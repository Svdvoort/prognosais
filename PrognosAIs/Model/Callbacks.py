import time

import tensorflow as tf


def calculate_concordance_index(y_true, y_pred):
    """
    This function determine the concordance index for two numpy arrays

    y_true contains a label to indicate whether events occurred, and time to events
    (or time to right censored data if no event occurred)

    y_pred is beta*x in the cox model
    """
    events_occurred = y_true[:, 0]
    time_to_events = y_true[:, 1]

    total_c = 0.0
    N = 0.0
    for i_event_occurred, i_time_to_event, i_h in zip(events_occurred, time_to_events, y_pred):
        if i_event_occurred == 1:
            for j_time_to_event, j_h in zip(time_to_events, y_pred):
                if j_time_to_event > i_time_to_event:
                    if i_h < j_h:
                        total_c += 1.0
                    N += 1.0

    concordance_index = total_c / N

    return concordance_index


class ConcordanceIndex(tf.keras.callbacks.Callback):
    """
    A custom callback function to evaluate the concordance index on the
    whole validation set
    """

    def __init__(self, validation_generator):
        self.validation_generator = validation_generator
        self.validation_labels = validation_generator.get_labels()
        return

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.validation_generator)
        y_real = self.validation_labels

        c_index = calculate_concordance_index(y_real, y_pred)

        logs["val_c_index"] = c_index
        return


class Timer(tf.keras.callbacks.Callback):
    """
    A custom callback function to evaluate the elapsed time of training
    """

    def __init__(self):
        self.start_time = time.time()
        return

    def on_epoch_end(self, epoch, logs=None):
        elapsed_time = time.time() - self.start_time

        logs["elapsed_time"] = elapsed_time
        return
