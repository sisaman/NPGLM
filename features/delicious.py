import logging
import os
import random
import threading
from datetime import datetime

import numpy as np
from scipy import sparse

from .utils import Indexer, create_sparse, timestamp_delta_generator


def extract_features(contact_sparse, save_sparse, attach_sparse, observed_samples, censored_samples):
    logging.info('extracting ...')
    num_metapaths = 6
    MP = [None for _ in range(num_metapaths)]
    events = [threading.Event() for _ in range(num_metapaths)]

    def worker(i):
        if i == 0:
            logging.debug('0: U-U-U')
            MP[i] = contact_sparse.dot(contact_sparse)
        elif i == 1:
            logging.debug('1: U-B-U')
            MP[i] = save_sparse.dot(save_sparse.T)
        elif i == 2:
            logging.debug('2: U-B-T-B-U')
            UBT = save_sparse.dot(attach_sparse.T)
            MP[i] = UBT.dot(UBT.T)
        elif i == 3:
            events[0].wait()
            logging.debug('3: U-U-U-U')
            MP[i] = MP[0].dot(contact_sparse)
        elif i == 4:
            events[1].wait()
            logging.debug('4: U-B-U-U')
            MP[i] = MP[1].dot(contact_sparse)
        elif i == 5:
            events[2].wait()
            logging.debug('5: U-B-T-B-U-U')
            MP[i] = MP[2].dot(contact_sparse)

        events[i].set()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_metapaths)]

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    def get_features(p, q):
        return [MP[i][p, q] for i in range(num_metapaths)]

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


def sample_generator(usr_dataset, observation_begin, observation_end, contact_sparse, indexer, censoring_ratio):
    logging.info('generating samples ...')
    U_U = contact_sparse.dot(contact_sparse.T)
    observed_samples = {}

    for line in usr_dataset[1:]:
        line_items = line.split('\t')
        contact_timestamp = float(line_items[2]) / 1000
        if observation_begin < contact_timestamp <= observation_end:
            u = indexer.get_index('user', line_items[0])
            v = indexer.get_index('user', line_items[1])
            if not (u is None or v is None):
                observed_samples[u, v] = contact_timestamp - observation_begin

    # logging.info('Observed samples found.')

    nonzero = sparse.find(U_U)
    set_observed = set([(u, v) for (u, v) in observed_samples] + [(u, v) for (u, v) in zip(nonzero[0], nonzero[1])])
    censored_samples = {}
    N = U_U.shape[0]
    M = len(observed_samples) // ((1 / censoring_ratio) - 1)
    user_list = [i for i in range(N)]

    while len(censored_samples) < M:
        i = random.randint(0, len(user_list) - 1)
        j = random.randint(0, len(user_list) - 1)
        if i != j:
            u = user_list[i]
            v = user_list[j]
            if (u, v) not in set_observed:
                censored_samples[u, v] = observation_end - observation_begin + 1

    # print(len(observed_samples) + len(censored_samples))

    return observed_samples, censored_samples


def generate_indexer(usr_dataset, usr_bm_tg, feature_begin, feature_end):
    logging.info('generating indexer ...')
    indexer = Indexer(['user', 'tag', 'bookmark'])
    min_time = 1e30
    max_time = -1

    for line in usr_dataset[1:]:
        line_items = line.split('\t')
        contact_timestamp = float(line_items[2]) / 1000
        min_time = min(min_time, contact_timestamp)
        max_time = max(max_time, contact_timestamp)
        if feature_begin < contact_timestamp <= feature_end:
            indexer.index('user', line_items[0])
            indexer.index('user', line_items[1])

    for line in usr_bm_tg[1:]:
        line_items = line.split('\t')
        tag_timestamp = float(line_items[3]) / 1000
        if feature_begin < tag_timestamp <= feature_end:
            indexer.index('user', line_items[0])
            indexer.index('bookmark', line_items[1])
            indexer.index('tag', line_items[2])

    with open('delicious/data/metadata.txt', 'w') as output:
        output.write('Nodes:\n')
        output.write('-----------------------------\n')
        output.write('#Users: %d\n' % indexer.indices['user'])
        output.write('#Tags: %d\n' % indexer.indices['tag'])
        output.write('#Bookmarks: %d\n' % indexer.indices['bookmark'])
        output.write('\nEdges:\n')
        output.write('-----------------------------\n')
        output.write('#Contact: %d\n' % len(usr_dataset))
        output.write('#Save : %d\n' % len(usr_bm_tg))
        output.write('#Attach: %d\n' % len(usr_bm_tg))
        output.write('\nTime Span:\n')
        output.write('-----------------------------\n')
        output.write('From: %s\n' % datetime.fromtimestamp(min_time))
        output.write('To: %s\n' % datetime.fromtimestamp(max_time))

    return indexer


def parse_dataset(usr_dataset, usr_bm_tg, feature_begin, feature_end, indexer):
    logging.info('parsing dataset ...')
    contact = []
    save = []
    attach = []

    # while parsing the users dataset we extract the contact relationships
    #  occurring between users in the feature extraction window
    for line in usr_dataset[1:]:  # skipping the first line (header) of the dataset
        line_items = line.split('\t')
        contact_timestamp = float(line_items[2]) / 1000

        if feature_begin < contact_timestamp <= feature_end:
            user1, user2 = (indexer.get_index('user', line_items[i]) for i in range(2))
            contact.append((user1, user2))

    # while parsing the user_tag_bookmark dataset we extract the relationships
    #  occurring between these entities in the feature extraction window
    for line in usr_bm_tg[1:]:
        line_items = line.split('\t')
        assign_time = float(line_items[3]) / 1000

        if feature_begin < assign_time <= feature_end:
            user = indexer.get_index('user', line_items[0])
            bookmark = indexer.get_index('bookmark', line_items[1])
            tag = indexer.get_index('tag', line_items[2])
            save.append((user, bookmark))
            attach.append((tag, bookmark))

    num_usr = indexer.indices['user']
    num_tag = indexer.indices['tag']
    num_bookmark = indexer.indices['bookmark']

    contact_sparse = create_sparse(contact, num_usr, num_usr)
    save_sparse = create_sparse(save, num_usr, num_bookmark)
    attach_sparse = create_sparse(attach, num_tag, num_bookmark)

    return contact_sparse, save_sparse, attach_sparse


def run(delta, observation_window, n_snapshots, censoring_ratio=0.5, single_snapshot=False):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt='%H:%M:%S')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cur_path = os.getcwd()
    os.chdir(dir_path)
    with open('delicious/data/user_contacts-timestamps.dat') as usr_usr:
        usr_dataset = usr_usr.read().splitlines()
    with open('delicious/data/user_taggedbookmarks-timestamps.dat') as usr_bm_tg:
        usr_bm_tg_dataset = usr_bm_tg.read().splitlines()

    delta = timestamp_delta_generator(months=delta)  # [1 2 3]
    # observation_window = 24  # [12 18 24]
    # n_snapshots = 15  # [9 12 15]

    observation_end = datetime(2010, 10, 1).timestamp()
    observation_begin = observation_end - timestamp_delta_generator(months=observation_window)
    feature_end = observation_begin
    feature_begin = feature_end - n_snapshots * delta

    # feature_begin = datetime(2009, 2, 1).timestamp()
    # feature_end = datetime(2009, 7, 1).timestamp()
    # observation_begin = datetime(2009, 7, 1).timestamp()
    # observation_end = datetime(2010, 1, 1).timestamp()

    # first we need to parse the whole data set to capture all of the entities and assign indexes to them
    indexer = generate_indexer(usr_dataset, usr_bm_tg_dataset, feature_begin, feature_end)

    # print(datetime.fromtimestamp(feature_end))
    # in this method we parse our dataset in the feature extraction window, and generate
    # the sparse matrices dedicated to each link
    contact_sparse, save_sparse, attach_sparse = parse_dataset(usr_dataset, usr_bm_tg_dataset,
                                                               feature_begin, feature_end, indexer)

    # in this method we would like to extract the target relationships that have been
    # generated in the observation window and after observation_end_time e.g. censored sample
    observed_samples, censored_samples = sample_generator(usr_dataset, observation_begin, observation_end,
                                                          contact_sparse, indexer, censoring_ratio)

    X, Y, T = extract_features(contact_sparse, save_sparse, attach_sparse, observed_samples, censored_samples)
    X_list = [X]

    # print(delta)
    # print(observation_end - observation_begin)
    if not single_snapshot:
        for t in range(int(feature_end - delta), int(feature_begin - 1), -int(delta)):
            # print(datetime.fromtimestamp(t))
            contact_sparse, save_sparse, attach_sparse = parse_dataset(
                usr_dataset, usr_bm_tg_dataset, feature_begin, t, indexer)
            X, _, _ = extract_features(contact_sparse, save_sparse, attach_sparse, observed_samples, censored_samples)
            X_list.append(X)

    # X = np.stack(X_list[::-1], axis=1)  # X.shape = (n_samples, timesteps, n_features)
    # pickle.dump({'X': X_list[::-1], 'Y': Y, 'T': T}, open('delicious/data/dataset.pkl', 'wb'))
    logging.info('done.')
    os.chdir(cur_path)
    return X_list, Y, T


if __name__ == '__main__':
    run(delta=1, observation_window=12, n_snapshots=9)
    # pass