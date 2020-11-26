from .base import Array, np, cp
from typing import Any, Union

def gaus_window(array: Array, sigma: float = 0.4) -> Array:
    ncp = np if isinstance(array, np.ndarray) else cp
    sfreq = array.shape[-1]
    x = ncp.arange(0, sfreq, 1) / sfreq - 0.5
    window = ncp.exp(-ncp.square(x/sigma))
    return array * window

def hanning_window(sfreq: int, hanning_ratio: float = 0.50,
                   tukey_ratio: float = 0,
                   blackman: Union[float, bool] = 0,
                   cuda: bool = False) -> Array:
    """
    Hanning windows family function, but it yields not only hanning window.
    Available windows are
    - Han window
    - Hamming window
    - Blackman window
    - Tukey window

    Not very fast, and so, making every time when analysing is not recommended.
    Just yield the matrix, and multiply.

    hanning_ratio: float
        Ratio of Han window and Hamming window.
        If 0.5, Han window.
        If 0.54, Hamming window.
        Default is 0.5.
    tukey_ratio: float
        Ratio of Tukey window.
        If it is 0, it becomes Han of Hamming window.
        Default is 0.
    blackman: Union[float, bool]
        Whether use blackman or not.
        If you set it True, Blackman window will be created.
        If you want to control the ratio of cos(4x), you can set float.
        Default is 0.
    """
    ncp = cp if cuda else np
    def mul(x: Any, y: Any) -> Any:
        return 0 if x == 0 else x * y
    blackman = 0.08 if blackman is True else blackman
    x = ncp.arange(0, sfreq, 1) / sfreq
    if tukey_ratio == 0:
        return (hanning_ratio\
            - mul(1 - hanning_ratio - blackman, ncp.cos(2 * ncp.pi * x)))\
            + mul(blackman, ncp.cos(4 * ncp.pi * x))
    else:
        x = ncp.where(
            (tukey_ratio / 2 < x) & (x < 1 - tukey_ratio / 2),
            1,
            x)
        x = ncp.where(
            x != 1,
            hanning_ratio
            + mul(1-hanning_ratio,
                  ncp.cos(2 * ncp.pi * (x - tukey_ratio/2) / tukey_ratio)),
            1)
        return x
