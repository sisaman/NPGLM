import bisect
import logging
import os
import random
import threading

import Stemmer
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords as stop_words
from scipy import sparse

from .utils import Indexer, create_sparse

path = 'th'
paper_threshold = 5

stopwords = stop_words.words('english')
stem = Stemmer.Stemmer('english')


class Paper:
    def __init__(self, year):
        self.id = None
        self.authors = []
        self.year = year
        self.venue = None
        self.references = []
        self.terms = []

    def __lt__(self, other):
        return self.year < other.year

    def __le__(self, other):
        return self.year <= other.year

    def __gt__(self, other):
        return self.year > other.year

    def __ge__(self, other):
        return self.year >= other.year


def parse_term(title):
    title = title.replace('-', ' ')
    title = title.replace(':', ' ')
    title = title.replace(';', ' ')
    wlist = title.strip().split()
    token = [j for j in wlist if j not in stopwords]
    token = stem.stemWords(token)
    return token


def generate_papers(datafile, feature_begin, feature_end, observation_begin, observation_end, conf_list):
    logging.info('generating papers ...')

    # try:
    #     result = pickle.load(open('dblp/data/papers_%s.pkl' % path, 'rb'))
    #     return result
    # except IOError:
    #     pass

    indexer = Indexer(['author', 'paper', 'term', 'venue'])

    index, authors, title, year, venue = None, None, None, None, None
    references = []

    write = 0
    cite = 0
    include = 0
    published = 0

    min_year = 3000
    max_year = 0

    papers_feature_window = []
    papers_observation_window = []

    with open(datafile) as file:
        dataset = file.read().splitlines()

    for line in dataset:
        if not line:
            if year and venue:
                year = int(year)
                if year > 0 and authors and venue in conf_list:
                    min_year = min(min_year, year)
                    max_year = max(max_year, year)
                    authors = authors.split(',')
                    terms = parse_term(title)
                    write += len(authors)
                    cite += len(references)
                    include += len(terms)
                    published += 1

                    p = Paper(year)
                    if feature_begin < year <= feature_end:
                        p.id = indexer.index('paper', index)
                        p.terms = [indexer.index('term', term) for term in terms]
                        p.references = [indexer.index('paper', paper_id) for paper_id in references]
                        p.authors = [indexer.index('author', author_name) for author_name in authors]
                        p.venue = indexer.index('venue', venue)
                        bisect.insort(papers_feature_window, p)
                    elif observation_begin < year <= observation_end:
                        p.references = references
                        p.authors = authors
                        papers_observation_window.append(p)

            index, authors, title, year, venue = None, None, None, None, None
            references = []
        else:
            begin = line[1]
            if begin == '*':
                title = line[2:]
            elif begin == '@':
                authors = line[2:]
            elif begin == 't':
                year = line[2:]
            elif begin == 'c':
                venue = line[2:]
            elif begin == 'i':
                index = line[6:]
            elif begin == '%':
                references.append(line[2:])

    for p in papers_observation_window:
        authors = []
        references = []
        for author in p.authors:
            author_id = indexer.get_index('author', author)
            if author_id is not None:
                authors.append(author_id)
        for ref in p.references:
            paper_id = indexer.get_index('paper', ref)
            if paper_id is not None:
                references.append(paper_id)
        p.authors = authors
        p.references = references

    with open('data/dblp/metadata_%s.txt' % path, 'w') as output:
        output.write('Nodes:\n')
        output.write('-----------------------------\n')
        output.write('#Authors: %d\n' % indexer.indices['author'])
        output.write('#Papers: %d\n' % indexer.indices['paper'])
        output.write('#Venues: %d\n' % indexer.indices['venue'])
        output.write('#Terms: %d\n\n' % indexer.indices['term'])
        output.write('\nEdges:\n')
        output.write('-----------------------------\n')
        output.write('#Write: %d\n' % write)
        output.write('#Cite: %d\n' % cite)
        output.write('#Publish: %d\n' % published)
        output.write('#Contain: %d\n' % include)
        output.write('\nTime Span:\n')
        output.write('-----------------------------\n')
        output.write('From: %s\n' % min_year)
        output.write('To: %s\n' % max_year)

    result = papers_feature_window, papers_observation_window, indexer.indices
    # pickle.dump(result, open('dblp/data/papers_%s.pkl' % path, 'wb'))
    return result


def parse_dataset(papers_feature_window, feature_begin, feature_end, counter):
    logging.info('parsing dataset ...')
    write = []
    cite = []
    include = []
    published = []

    left_gt = Paper(feature_begin)
    right_le = Paper(feature_end)
    left_index = bisect.bisect_right(papers_feature_window, left_gt)
    right_index = bisect.bisect_right(papers_feature_window, right_le)

    for p in papers_feature_window[left_index:right_index]:
        for author_id in p.authors:
            write.append((author_id, p.id))
        for paper_id in p.references:
            cite.append((paper_id, p.id))
        for term_id in p.terms:
            include.append((p.id, term_id))
        published.append((p.id, p.venue))

    num_authors = counter['author']
    num_papers = counter['paper']
    num_venues = counter['venue']
    num_terms = counter['term']

    W = create_sparse(write, num_authors, num_papers)
    C = create_sparse(cite, num_papers, num_papers)
    I = create_sparse(include, num_papers, num_terms)
    P = create_sparse(published, num_papers, num_venues)

    return W, C, I, P


def extract_features(W, C, P, I, observed_samples, censored_samples):
    logging.info('extracting ...')
    MP = [None for _ in range(24)]
    events = [threading.Event() for _ in range(24)]

    def worker(i):
        if i == 0:
            logging.debug('0: A-P-A')
            MP[i] = W.dot(W.T)
        elif i == 1:
            events[0].wait()
            logging.debug('1: A-P-A-P-A')
            MP[i] = MP[0].dot(MP[0].T)
        elif i == 2:
            events[19].wait()
            logging.debug('2: A-P-V-P-A')
            MP[i] = MP[19].dot(MP[19].T)
        elif i == 3:
            events[20].wait()
            logging.debug('3: A-P-T-P-A')
            MP[i] = MP[20].dot(MP[20].T)
        elif i == 4:
            events[21].wait()
            logging.debug('4: A-P->P<-P-A')
            MP[i] = MP[21].dot(MP[21].T)
        elif i == 5:
            events[22].wait()
            logging.debug('5: A-P<-P->P-A')
            MP[i] = MP[22].dot(MP[22].T)
        elif i == 6:
            events[21].wait()
            events[22].wait()
            logging.debug('6: A-P->P->P-A')
            MP[i] = MP[21].dot(MP[22].T)
        elif i == 7:
            events[0].wait()
            events[23].wait()
            logging.debug('7: A-P-P-A-P-A')
            MP[i] = MP[23].dot(MP[0])
        elif i == 8:
            events[1].wait()
            events[23].wait()
            logging.debug('8: A-P-P-A-P-A-P-A')
            MP[i] = MP[23].dot(MP[1])
        elif i == 9:
            events[2].wait()
            events[23].wait()
            logging.debug('9: A-P-P-A-P-V-P-A')
            MP[i] = MP[23].dot(MP[2])
        elif i == 10:
            events[3].wait()
            events[23].wait()
            logging.debug('10: A-P-P-A-P-T-P-A')
            MP[i] = MP[23].dot(MP[3])
        elif i == 11:
            events[4].wait()
            events[23].wait()
            logging.debug('11: A-P-P-A-P->P<-P-A')
            MP[i] = MP[23].dot(MP[4])
        elif i == 12:
            events[5].wait()
            events[23].wait()
            logging.debug('12: A-P-P-A-P<-P->P-A')
            MP[i] = MP[23].dot(MP[5])
        elif i == 13:
            events[0].wait()
            events[23].wait()
            logging.debug('13: A-P-A-P-P-A')
            MP[i] = MP[0].dot(MP[23])
        elif i == 14:
            events[1].wait()
            events[23].wait()
            logging.debug('14: A-P-A-P-A-P-P-A')
            MP[i] = MP[1].dot(MP[23])
        elif i == 15:
            events[2].wait()
            events[23].wait()
            logging.debug('15: A-P-V-P-A-P-P-A')
            MP[i] = MP[2].dot(MP[23])
        elif i == 16:
            events[3].wait()
            events[23].wait()
            logging.debug('16: A-P-T-P-A-P-P-A')
            MP[i] = MP[3].dot(MP[23])
        elif i == 17:
            events[4].wait()
            events[23].wait()
            logging.debug('17: A-P->P<-P-A-P-P-A')
            MP[i] = MP[4].dot(MP[23])
        elif i == 18:
            events[5].wait()
            events[23].wait()
            logging.debug('18: A-P<-P->P-A-P-P-A')
            MP[i] = MP[5].dot(MP[23])
        elif i == 19:
            logging.debug('A-P-V')
            MP[i] = W.dot(P)
        elif i == 20:
            logging.debug('A-P-T')
            MP[i] = W.dot(I)
        elif i == 21:
            logging.debug('A-P->P')
            MP[i] = W.dot(C)
        elif i == 22:
            logging.debug('A-P<-P')
            MP[i] = W.dot(C.T)
        elif i == 23:
            events[21].wait()
            logging.debug('A-P-P-A')
            MP[23] = MP[21].dot(W.T)

        events[i].set()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(24)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    def get_features(p, q):
        return [MP[i][p, q] for i in range(19)]

    X = []
    Y = []
    T = []

    for (u, v) in observed_samples:
        t = observed_samples[u, v]
        fv = get_features(u, v)
        X.append(fv)
        Y.append(True)
        T.append(t)

    for (u, v) in censored_samples:
        t = censored_samples[u, v]
        fv = get_features(u, v)
        X.append(fv)
        Y.append(False)
        T.append(t)

    return np.array(X), np.array(Y), np.array(T)


def generate_samples(papers_observation_window, censoring_ratio, W, C):
    logging.info('generating samples ...')
    written_by = {}
    elements = sparse.find(W)
    for i in range(len(elements[0])):
        author = elements[0][i]
        paper = elements[1][i]
        if paper in written_by:
            written_by[paper].append(author)
        else:
            written_by[paper] = [author]

    APPA = W.dot(C.dot(W.T))
    num_papers = (W.dot(W.T)).diagonal()
    observed_samples = {}

    for p in papers_observation_window:
        for u in p.authors:
            if num_papers[u] >= paper_threshold:
                for paper_id in p.references:
                    if paper_id in written_by:
                        for v in written_by[paper_id]:
                            if num_papers[v] >= paper_threshold and not APPA[u, v]:
                                if (u, v) in observed_samples:
                                    observed_samples[u, v] = min(p.year, observed_samples[u, v])
                                else:
                                    observed_samples[u, v] = p.year

    # logging.info('Observed samples found.')
    nonzero = sparse.find(APPA)
    set_observed = set([(u, v) for (u, v) in observed_samples] + [(u, v) for (u, v) in zip(nonzero[0], nonzero[1])])
    censored_samples = {}
    N = APPA.shape[0]
    M = len(observed_samples) // ((1 / censoring_ratio) - 1)
    author_list = [i for i in range(N) if num_papers[i] >= paper_threshold]

    while len(censored_samples) < M:
        i = random.randint(0, len(author_list) - 1)
        j = random.randint(0, len(author_list) - 1)
        if i != j:
            u = author_list[i]
            v = author_list[j]
            if (u, v) not in set_observed:
                censored_samples[u, v] = papers_observation_window[-1].year + 1

    # print(len(observed_samples) + len(censored_samples))
    return observed_samples, censored_samples


def run(delta, observation_window, n_snapshots, censoring_ratio=0.5, single_snapshot=False):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    conf_list = {
        'db': [
            'KDD', 'PKDD', 'ICDM', 'SDM', 'PAKDD', 'SIGMOD', 'VLDB', 'ICDE', 'PODS', 'EDBT', 'SIGIR', 'ECIR',
            'ACL', 'WWW', 'CIKM', 'NIPS', 'ICML', 'ECML', 'AAAI', 'IJCAI',
        ],
        'th': [
            'STOC', 'FOCS', 'COLT', 'LICS', 'SCG', 'SODA', 'SPAA', 'PODC', 'ISSAC', 'CRYPTO', 'EUROCRYPT', 'CONCUR',
            'ICALP', 'STACS', 'COCO', 'WADS', 'MFCS', 'SWAT', 'ESA', 'IPCO', 'LFCS', 'ALT', 'EUROCOLT', 'WDAG', 'ISTCS',
            'FSTTCS', 'LATIN', 'RECOMB', 'CADE', 'ISIT', 'MEGA', 'ASIAN', 'CCCG', 'FCT', 'WG', 'CIAC', 'ICCI', 'CATS',
            'COCOON', 'GD', 'ISAAC', 'SIROCCO', 'WEA', 'ALENEX', 'FTP', 'CSL', 'DMTCS'
        ]
    }[path]

    # delta = 1
    # observation_window = 3
    # n_snapshots = 5

    observation_end = 2016
    observation_begin = observation_end - observation_window
    feature_end = observation_begin
    feature_begin = feature_end - delta * n_snapshots

    papers_feat_window, papers_obs_window, counter = generate_papers('data/dblp/dblp.txt', feature_begin, feature_end,
                                                                     observation_begin, observation_end,
                                                                     conf_list)
    W, C, I, P = parse_dataset(papers_feat_window, feature_begin, feature_end, counter)
    observed_samples, censored_samples = generate_samples(papers_obs_window, censoring_ratio, W, C)

    X, Y, T = extract_features(W, C, P, I, observed_samples, censored_samples)
    T -= observation_begin
    X_list = [X]

    if not single_snapshot:

        for t in range(feature_end - delta, feature_begin - 1, -delta):
            # print('=============%d=============' % t)
            W, C, I, P = parse_dataset(papers_feat_window, feature_begin, t, counter)
            X, _, _ = extract_features(W, C, P, I, observed_samples, censored_samples)
            X_list.append(X)

    # X = np.stack(X_list[::-1], axis=1)  # X.shape = (n_samples, timesteps, n_features)
    # pickle.dump({'X': X_list[::-1], 'Y': Y, 'T': T}, open('dblp/data/dataset_%s.pkl' % path, 'wb'))
    logging.info('done.')
    return X_list, Y, T


if __name__ == '__main__':
    # main()
    run(1, 6, 12)
    # print('success')
    pass
