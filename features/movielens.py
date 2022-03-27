import logging
import os
import random
import threading
from datetime import datetime

import numpy as np
from scipy import sparse

from .utils import Indexer, create_sparse, timestamp_delta_generator

rating_threshold = 4
actor_threshold = 3


def generate_indexer(user_rates_movies_ds, user_tags_movies_ds, movie_actor_ds,
                     movie_director_ds, movie_genre_ds, movie_countries_ds, feature_begin, feature_end):
    logging.info('generating indexer ...')
    min_time = 1e30
    max_time = -1
    indexer = Indexer(['user', 'tag', 'movie', 'actor', 'director', 'genre', 'country'])

    for line in user_rates_movies_ds[1:]:
        line_items = line.split('\t')
        rating_timestamp = float(line_items[3]) / 1000
        min_time = min(min_time, rating_timestamp)
        max_time = max(max_time, rating_timestamp)
        rating = float(line_items[2])
        if feature_begin < rating_timestamp <= feature_end and rating > rating_threshold:
            indexer.index('user', line_items[0])
            indexer.index('movie', line_items[1])

    for line in user_tags_movies_ds[1:]:
        line_items = line.split('\t')
        tag_timestamp = float(line_items[3]) / 1000
        if feature_begin < tag_timestamp <= feature_end:
            indexer.index('user', line_items[0])
            indexer.index('movie', line_items[1])
            indexer.index('tag', line_items[2])

    for line in movie_actor_ds[1:]:
        line_items = line.split('\t')
        ranking = int(line_items[3])
        if ranking < actor_threshold and line_items[0] in indexer.mapping['movie']:
            # indexer.index('movie', line_items[0])
            indexer.index('actor', line_items[1])

    for line in movie_director_ds[1:]:
        line_items = line.split('\t')
        if line_items[0] in indexer.mapping['movie']:
            # indexer.index('movie', line_items[0])
            indexer.index('director', line_items[1])

    for line in movie_genre_ds[1:]:
        line_items = line.split('\t')
        if line_items[0] in indexer.mapping['movie']:
            # indexer.index('movie', line_items[0])
            indexer.index('genre', line_items[1])

    for line in movie_countries_ds[1:]:
        line_items = line.split('\t')
        if line_items[0] in indexer.mapping['movie']:
            # indexer.index('movie', line_items[0])
            indexer.index('country', line_items[1])

    with open('data/movielens/metadata.txt', 'w') as output:
        output.write('Nodes:\n')
        output.write('-----------------------------\n')
        output.write('#Users: %d\n' % indexer.indices['user'])
        output.write('#Tags: %d\n' % indexer.indices['tag'])
        output.write('#Movies: %d\n' % indexer.indices['movie'])
        output.write('#Actors: %d\n' % indexer.indices['actor'])
        output.write('#Director: %d\n' % indexer.indices['director'])
        output.write('#Genre: %d\n' % indexer.indices['genre'])
        output.write('#Countriy: %d\n' % indexer.indices['country'])
        output.write('\nEdges:\n')
        output.write('-----------------------------\n')
        output.write('#Rate: %d\n' % len(user_rates_movies_ds))
        output.write('#Attach: %d\n' % len(user_tags_movies_ds))
        output.write('#Played_by: %d\n' % len(movie_actor_ds))
        output.write('#Directed_by : %d\n' % len(movie_director_ds))
        output.write('#Has: %d\n' % len(movie_genre_ds))
        output.write('#Produced_in: %d\n' % len(movie_countries_ds))
        output.write('\nTime Span:\n')
        output.write('-----------------------------\n')
        output.write('From: %s\n' % datetime.fromtimestamp(min_time))
        output.write('To: %s\n' % datetime.fromtimestamp(max_time))

    return indexer


def parse_dataset(user_rates_movies_ds,
                  user_tags_movies_ds, movie_actor_ds, movie_director_ds, movie_genre_ds,
                  movie_countries_ds, feature_begin, feature_end, indexer):
    logging.info('parsing dataset ...')
    rate = []
    # assign = []
    attach = []
    played_by = []
    directed_by = []
    has = []
    produced_in = []

    # while parsing the users dataset we extract the contact relationships
    #  occurring between users in the feature extraction window
    for line in user_rates_movies_ds[1:]:  # skipping the first line (header) of the dataset
        line_items = line.split('\t')
        # the timestamp int he dataset is represented with miliseconds, so
        # we eliminate the last 3 charactars
        rating = float(line_items[2])
        rating_timestamp = float(line_items[3]) / 1000
        if feature_begin < rating_timestamp <= feature_end and rating > rating_threshold:
            user = indexer.get_index('user', line_items[0])
            movie = indexer.get_index('movie', line_items[1])
            rate.append((user, movie))

    # while parsing the user_tag_bookmark dataset we extract the relationships
    #  occurring between these entities in the feature extraction window
    for line in user_tags_movies_ds[1:]:
        line_items = line.split('\t')
        assign_time = float(line_items[3]) / 1000
        if feature_begin < assign_time <= feature_end:
            # user = indexer.get_index('user', line_items[0])
            movie = indexer.get_index('movie', line_items[1])
            tag = indexer.get_index('tag', line_items[2])
            # assign.append((user, tag))
            attach.append((tag, movie))

    for line in movie_actor_ds[1:]:
        line_items = line.split('\t')
        ranking = int(line_items[3])
        if ranking < actor_threshold:
            movie = indexer.get_index('movie', line_items[0])
            actor = indexer.get_index('actor', line_items[1])
            if not (movie is None or actor is None):
                played_by.append((movie, actor))

    for line in movie_director_ds[1:]:
        line_items = line.split('\t')
        movie = indexer.get_index('movie', line_items[0])
        director = indexer.get_index('director', line_items[1])
        if not (movie is None or director is None):
            directed_by.append((movie, director))

    for line in movie_genre_ds[1:]:
        line_items = line.split('\t')
        movie = indexer.get_index('movie', line_items[0])
        genre = indexer.get_index('genre', line_items[1])
        if not (movie is None or genre is None):
            has.append((movie, genre))

    for line in movie_countries_ds[1:]:
        line_items = line.split('\t')
        movie = indexer.get_index('movie', line_items[0])
        country = indexer.get_index('country', line_items[1])
        if not (movie is None or country is None):
            produced_in.append((movie, country))

    num_usr = indexer.indices['user']
    num_tag = indexer.indices['tag']
    num_movie = indexer.indices['movie']
    num_actor = indexer.indices['actor']
    num_directors = indexer.indices['director']
    num_genre = indexer.indices['genre']
    num_countries = indexer.indices['country']

    rate_sparse = create_sparse(rate, num_usr, num_movie)
    # assign_sparse = create_sparse(assign, num_usr, num_tag)
    attach_sparse = create_sparse(attach, num_tag, num_movie)
    played_by_sparse = create_sparse(played_by, num_movie, num_actor)
    directed_by_sparse = create_sparse(directed_by, num_movie, num_directors)
    has_genre_sparse = create_sparse(has, num_movie, num_genre)
    produced_in_sparse = create_sparse(produced_in, num_movie, num_countries)

    return rate_sparse, attach_sparse, played_by_sparse, directed_by_sparse, has_genre_sparse, produced_in_sparse
    # assign_sparse


def sample_generator(usr_rates_movies_ds, observation_begin, observation_end, rate_sparse, indexer, censoring_ratio):
    logging.info('generating samples ...')
    U_M = rate_sparse
    observed_samples = {}

    for line in usr_rates_movies_ds[1:]:
        line_items = line.split('\t')
        rating = float(line_items[2])
        rating_timestamp = float(line_items[3]) / 1000
        if observation_begin < rating_timestamp <= observation_end and rating > rating_threshold:
            u = indexer.get_index('user', line_items[0])
            v = indexer.get_index('movie', line_items[1])
            if not (u is None or v is None):
                observed_samples[u, v] = rating_timestamp - observation_begin

    # logging.info('Observed samples found.')

    nonzero = sparse.find(U_M)
    set_observed = set([(u, v) for (u, v) in observed_samples] + [(u, v) for (u, v) in zip(nonzero[0], nonzero[1])])
    censored_samples = {}

    M = len(observed_samples) // ((1 / censoring_ratio) - 1)
    user_list = [i for i in range(U_M.shape[0])]
    movie_list = [i for i in range(U_M.shape[1])]

    while len(censored_samples) < M:
        i = random.randint(0, len(user_list) - 1)
        j = random.randint(0, len(movie_list) - 1)
        if i != j:
            u = user_list[i]
            v = movie_list[j]
            if (u, v) not in set_observed:
                censored_samples[u, v] = observation_end - observation_begin + 1

    # print(len(observed_samples) + len(censored_samples))
    return observed_samples, censored_samples


def extract_features(rate_sparse, attach_sparse, played_by_sparse, directed_by_sparse,
                     has_genre_sparse, produced_in_sparse, observed_samples, censored_samples):
    logging.info('extracting ...')
    num_metapaths = 11
    MP = [None for _ in range(num_metapaths)]
    events = [threading.Event() for _ in range(num_metapaths)]
    MUM_sparse = rate_sparse.T.dot(rate_sparse)

    def worker(i):
        if i == 0:
            MP[i] = rate_sparse.dot(played_by_sparse.dot(played_by_sparse.T))
            logging.debug('0: U-M-A-M')
        elif i == 1:
            MP[i] = rate_sparse.dot(directed_by_sparse.dot(directed_by_sparse.T))
            logging.debug('1: U-M-D-M')
        elif i == 2:
            MP[i] = rate_sparse.dot(has_genre_sparse.dot(has_genre_sparse.T))
            logging.debug('2: U-M-G-M')
        elif i == 3:
            MP[i] = rate_sparse.dot(attach_sparse.T.dot(attach_sparse))
            logging.debug('3: U-M-T-M')
        elif i == 4:
            MP[i] = rate_sparse.dot(produced_in_sparse.dot(produced_in_sparse.T))
            logging.debug('4: U-M-C-M')
        elif i == 5:
            MP[i] = rate_sparse.dot(MUM_sparse)
            logging.debug('5: U-M-U-M')
        elif i == 6:
            events[0].wait()
            MP[i] = MP[0].dot(MUM_sparse)
            logging.debug('6: U-M-A-M-U-M')
        elif i == 7:
            events[1].wait()
            MP[i] = MP[1].dot(MUM_sparse)
            logging.debug('7: U-M-D-M-U-M')
        elif i == 8:
            events[2].wait()
            MP[i] = MP[2].dot(MUM_sparse)
            logging.debug('8: U-M-G-M-U-M')
        elif i == 9:
            events[3].wait()
            MP[i] = MP[3].dot(MUM_sparse)
            logging.debug('10: U-M-T-M-U-M')
        elif i == 10:
            events[4].wait()
            MP[i] = MP[4].dot(MUM_sparse)
            logging.debug('9: U-M-C-M-U-M')
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


def run(delta, observation_window, n_snapshots, censoring_ratio=0.5, single_snapshot=False):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cur_path = os.getcwd()
    os.chdir(dir_path)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', datefmt='%H:%M:%S')

    with open('data/movielens/user_ratedmovies-timestamps.dat') as user_rates_movies_ds:
        user_rates_movies_ds = user_rates_movies_ds.read().splitlines()
    with open('data/movielens/user_taggedmovies-timestamps.dat') as user_tags_movies_ds:
        user_tags_movies_ds = user_tags_movies_ds.read().splitlines()
    with open('data/movielens/movie_actors.dat', encoding='latin-1') as movie_actor_ds:
        movie_actor_ds = movie_actor_ds.read().splitlines()
    with open('data/movielens/movie_directors.dat', encoding='latin-1') as movie_director_ds:
        movie_director_ds = movie_director_ds.read().splitlines()
    with open('data/movielens/movie_genres.dat') as movie_genre_ds:
        movie_genre_ds = movie_genre_ds.read().splitlines()
    with open('data/movielens/movie_countries.dat') as movie_countries_ds:
        movie_countries_ds = movie_countries_ds.read().splitlines()

    delta = timestamp_delta_generator(months=delta)  # [1 2 3]
    # observation_window = 24  # [12 18 24]
    # n_snapshots = 15  # [9 12 15]

    observation_end = datetime(2009, 1, 1).timestamp()
    observation_begin = observation_end - timestamp_delta_generator(months=observation_window)
    feature_end = observation_begin
    feature_begin = feature_end - n_snapshots * delta

    # feature_begin = datetime(2006, 1, 1).timestamp()
    # feature_end = datetime(2008, 1, 1).timestamp()
    # observation_begin = datetime(2008, 1, 1).timestamp()
    # observation_end = datetime(2009, 1, 1).timestamp()

    indexer = generate_indexer(user_rates_movies_ds, user_tags_movies_ds, movie_actor_ds,
                               movie_director_ds, movie_genre_ds, movie_countries_ds, feature_begin, feature_end)
    rate_sparse, attach_sparse, played_by_sparse, directed_by_sparse, has_genre_sparse, produced_in_sparse \
        = parse_dataset(
        user_rates_movies_ds,
        user_tags_movies_ds, movie_actor_ds, movie_director_ds,
        movie_genre_ds,
        movie_countries_ds, feature_begin, feature_end, indexer
    )
    observed_samples, censored_samples = sample_generator(user_rates_movies_ds, observation_begin,
                                                          observation_end, rate_sparse, indexer, censoring_ratio)
    X, Y, T = extract_features(rate_sparse, attach_sparse, played_by_sparse, directed_by_sparse,
                               has_genre_sparse, produced_in_sparse, observed_samples, censored_samples)
    X_list = [X]

    # print(delta)
    # print(observation_end - observation_begin)

    if not single_snapshot:

        for t in range(int(feature_end - delta), int(feature_begin - 1), -int(delta)):
            # print(datetime.fromtimestamp(t))
            # print(datetime.fromtimestamp(t))
            rate_sparse, attach_sparse, played_by_sparse, directed_by_sparse, has_genre_sparse, produced_in_sparse \
                = parse_dataset(
                user_rates_movies_ds,
                user_tags_movies_ds, movie_actor_ds, movie_director_ds,
                movie_genre_ds,
                movie_countries_ds, feature_begin, t, indexer
            )
            X, _, _ = extract_features(rate_sparse, attach_sparse, played_by_sparse, directed_by_sparse,
                                       has_genre_sparse, produced_in_sparse, observed_samples, censored_samples)
            X_list.append(X)

    # X = np.stack(X_list[::-1], axis=1)  # X.shape = (n_samples, timesteps, n_features)
    # pickle.dump({'X': X_list[::-1], 'Y': Y, 'T': T}, open('data/movielensset.pkl', 'wb'))
    logging.info('done.')
    os.chdir(cur_path)
    return X_list, Y, T


if __name__ == '__main__':
    run(delta=1, observation_window=12, n_snapshots=9)
    # pass
