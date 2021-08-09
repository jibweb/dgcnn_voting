from utils.logger import logd
from math import ceil
from multiprocessing import Pool, TimeoutError
import numpy as np
from numpy.random import choice
from sklearn.utils import shuffle


def dummy_fn(fn):
    return fn


class Dataset(object):
    '''
        Utilities function for reading dataset data
        fullset should be a list (one entry per class) of list
        (containing instances)
    '''
    def __init__(self,
                 batch_size=100,
                 balance_train_set=True,
                 balance_test_set=False,
                 balance_val_set=False,
                 shuffle_class=True,
                 one_hot=True,
                 val_set_pct=0.1):
        self.batch_size = batch_size
        self.balance_train_set = balance_train_set
        self.balance_test_set = balance_test_set
        self.balance_val_set = balance_val_set
        self.shuffle_class = shuffle_class
        self.one_hot = one_hot
        self.val_set_pct = val_set_pct
        self.pool = Pool(8)
        self.unprepared = True

    def _to_one_hot(self, idx):
        return [0]*(idx) + [1] + [0]*(len(self.class_dict)-idx-1)

    def _balance_by_oversampling(self, fullset):
        class_hist = [len(cls_fns) for cls_fns in fullset]
        max_class_elt = max(class_hist)
        for cls_idx, cls_fns in enumerate(fullset):
            elt_to_add = max_class_elt - class_hist[cls_idx]
            fullset[cls_idx] += list(np.array(cls_fns)[choice(len(cls_fns),
                                                              elt_to_add)])

            assert max_class_elt == len(fullset[cls_idx]),\
                "Failed to balance class {}".format(cls_idx)
        return fullset

    def _get_x_y_from_fullset(self, fullset):
        # Get the corresponding label vector
        cur_hist = [len(cls_fns) for cls_fns in fullset]
        y = [[cls_idx]*cls_elt_no
             for cls_idx, cls_elt_no in enumerate(cur_hist)]

        # Flatten the set
        x = [item for sublist in fullset for item in sublist]
        y = [item for sublist in y for item in sublist]

        if self.one_hot:
            y = [self._to_one_hot(y_val) for y_val in y]

        assert len(x) == len(y)

        return x, y

    def _get_set_sizes(self):
        # set_size = len(self.train_x)
        # self.train_batch_no = int(ceil(float(set_size) / self.batch_size))
        self.train_batch_no = len(self.train_x) / self.batch_size

        # set_size = len(self.val_x)
        # self.val_batch_no = int(ceil(float(set_size) / self.batch_size))
        self.val_batch_no = len(self.val_x) / self.batch_size

        # set_size = len(self.test_x)
        # self.test_batch_no = int(ceil(float(set_size) / self.batch_size))
        self.test_batch_no = len(self.test_x) / self.batch_size

    def prepare_sets(self):
        '''
        '''
        bundled_sets = self.get_train_test_dataset()
        if len(bundled_sets) == 2:
            self.train_set, self.test_set = bundled_sets

            # === Validation set preparation ==================================
            # We don't want the val set to always be the same
            self.train_set = [shuffle(sublist) for sublist in self.train_set]

            # Split the sets pre-balancing to make sure they are different
            elt_in_val_set = [int(len(cls_fns)*self.val_set_pct)
                              for cls_fns in self.train_set]

            self.val_set = [cls_fns[:elt_in_set] for cls_fns, elt_in_set
                            in zip(self.train_set, elt_in_val_set)]
            self.train_set = [cls_fns[elt_not_in_set:]
                              for cls_fns, elt_not_in_set
                              in zip(self.train_set, elt_in_val_set)]
        elif len(bundled_sets) == 3:
            self.train_set, self.test_set, self.val_set = bundled_sets

        # === Balance the sets ================================================
        if self.balance_train_set:
            self.train_set = self._balance_by_oversampling(self.train_set)

        if self.balance_test_set:
            self.test_set = self._balance_by_oversampling(self.test_set)

        if self.balance_val_set:
            self.val_set = self._balance_by_oversampling(self.val_set)

        # self.train_set = self._balance_by_oversampling(self.train_set)

        self.train_x, self.train_y = self._get_x_y_from_fullset(self.train_set)
        self.val_x, self.val_y = self._get_x_y_from_fullset(self.val_set)
        self.test_x, self.test_y = self._get_x_y_from_fullset(self.test_set)

        self._get_set_sizes()

        logd("Training set size: {}", len(self.train_x))
        logd("Validation set size: {}", len(self.val_x))
        logd("Test set size: {}", len(self.test_x))
        self.unprepared = False

    def batch_idx_generator(self, batch_no):
        '''
        '''
        return ((self.batch_size*batch_iter, self.batch_size*(batch_iter+1))
                for batch_iter in range(batch_no))

    def batch_generator(self, x, y, batch_no, process_fn, timeout):
        '''
        '''
        batch_idx_gen = self.batch_idx_generator(batch_no)
        x, y = shuffle(x, y)

        # Compute the features of the first batch
        b_start, b_end = batch_idx_gen.next()
        x_batch = self.pool.map(process_fn,
                                x[b_start:b_end])
        y_batch = y[b_start:b_end]

        for b_start, b_end in batch_idx_gen:
            # Asynchronously compute the next batch
            r_map = self.pool.map_async(process_fn, x[b_start:b_end])
            yield x_batch, y_batch
            try:
                x_batch = r_map.get(timeout=timeout)
            except TimeoutError as e:
                print e
                print "Error on", x[b_start:b_end]

            y_batch = y[b_start:b_end]
        # Yield the last batch
        yield x_batch, y_batch

    def train_batch(self, process_fn=dummy_fn, timeout=10):
        if self.unprepared:
            self.prepare_sets()

        return self.batch_generator(self.train_x,
                                    self.train_y,
                                    self.train_batch_no,
                                    process_fn,
                                    timeout)

    def val_batch(self, process_fn=dummy_fn, timeout=10):
        if self.unprepared:
            self.prepare_sets()

        return self.batch_generator(self.val_x,
                                    self.val_y,
                                    self.val_batch_no,
                                    process_fn,
                                    timeout)

    def test_batch(self, process_fn=dummy_fn, timeout=10):
        if self.unprepared:
            self.prepare_sets()

        return self.batch_generator(self.test_x,
                                    self.test_y,
                                    self.test_batch_no,
                                    process_fn,
                                    timeout)

    def close(self):
        self.pool.close()

    def get_train_test_dataset(self):
        ''' The one method you need to implement for each dataset
            It should simply return two list of list:
            *  each element of the list is a class
            *  each element of the sublist is a filename corresponding
               to an element of the given class
            In order, the first list of list is the training set, the second is
            the test set. One can be empty?

        '''
        raise NotImplementedError("This dataset does not provide a method to"
                                  " access the data !")
