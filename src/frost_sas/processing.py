from scipy.signal import correlate


def match_filter(rx_data, tx_data):

    matched_data = correlate(
        rx_data, tx_data, mode="same"
    )  # TODO probably needs some work

    return matched_data
