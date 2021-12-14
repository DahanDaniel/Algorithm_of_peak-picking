from typing import Union, List, TypeVar

import numpy as np

Num = Union[int, float]
R = TypeVar("R", bound="Radon")


class Radon:
    frequency: List[Num]
    amplitude: List[Num]
    damping: List[Num]
    speed: List[Num]

    s: int  # number of series
    n: int  # number of spectral points
    snr: Num or None

    dwmin: Num  # Lower limit of the speed dimension
    dwmax: Num  # Upper limit of the speed dimension
    ddw: Num  # Resolution step in the speed dimension

    fid: np.ndarray
    spectra: np.ndarray
    radon_spectra: np.ndarray

    def __init__(
        self,
        frequency,
        amplitude,
        damping,
        speed,
        s,
        n,
        dwmin,
        dwmax,
        ddw,
        snr=None,
    ):
        self.frequency = frequency
        self.amplitude = amplitude
        self.damping = damping
        self.speed = speed

        self.s = s
        self.n = n
        self.snr = snr

        self.dwmin = dwmin
        self.dwmax = dwmax
        self.ddw = ddw

        self.fid = self.create_fid()
        self.spectra = np.fft.fft(self.fid)
        self.radon_spectra = self.radon_transform()

    def create_fid(self) -> np.ndarray:
        fid = np.zeros((self.s, self.n), dtype="complex64")
        t = np.linspace(0, 1, self.n, endpoint=False)
        for i in range(self.s):
            for k in range(np.shape(self.amplitude)[0]):
                total_frequency = self.frequency[k] + i * self.speed[k]
                if (
                    0 <= total_frequency <= self.n
                ):  # Mute frequencies higher than the resolution.
                    exponent = (
                        2
                        * np.pi
                        * 1j
                        * (total_frequency + 1j * self.damping[k])
                        * t
                    )
                    fid[i] = np.add(
                        fid[i],
                        self.amplitude[k] * np.e ** exponent,
                    )
                if self.snr:  # Add random noise, skip first point.
                    noise = (
                        (1 / self.snr)
                        * np.max(self.amplitude)
                        * np.random.uniform(0, 1, self.n - 1)
                    )
                    fid[i][1 : self.n] = np.add(fid[i][1 : self.n], noise)
                    fid[i] = np.add(fid[i], -1 * np.average(noise))

        # Fix the first point for the Fast Fourier Transform.
        for i in range(self.s):
            fid[i][0] /= 2

        return fid

    def radon_transform(self) -> np.ndarray:
        s, n = np.shape(self.fid)

        # Phase correction
        dw_arr = np.arange(
            self.dwmin, self.dwmax, self.ddw
        )  # domain of rates of change
        p = np.zeros((len(dw_arr), *np.shape(self.fid)), dtype="complex64")
        a = 2 * np.pi * 1j * np.linspace(0, 1, n, endpoint=False)
        for i in range(len(dw_arr)):
            b = a * dw_arr[i]
            for k in range(s):
                p[i][k] = self.fid[k] * np.e ** (-b * k)

        # "Diagonal" summation
        pr = np.zeros((len(dw_arr), n), dtype="complex64")
        for i in range(len(dw_arr)):
            for j in range(s):
                pr[i] += p[i][j]

        # Fourier Transform
        radon_spectra = np.fft.fft(pr)

        return radon_spectra

    def generate_random_data(ranges_dict: dict) -> R:
        """
        The ranges_dict variable should contain the following Radon
        class fields' names strings as keys:
            ["frequency", "amplitude", "damping", "speed",
            "s", "n", "dwmin", "dwmax", "ddw", "snr"]
        and corresponding tuples specifying their ranges as values, as
        well as a "number_of_peaks" key with an analogous tuple of ints.

        If you want to set a specific value to constant just pass the
        range tuple with lower_bound == upper_bound, or
        lower_bound == upper_bound - 1 in case of "s", "n" and
        "number_of_peaks".

        General key-value pair form:
        "field_name': (lower_bound, upper_bound)

        If snr is not specified or set to 0, then no noise is added.
        """
        # Draw a random number of peaks.
        N = np.random.randint(*ranges_dict["number_of_peaks"])

        # Randomize peaks' parameters.
        amplitude = np.random.uniform(*ranges_dict["amplitude"], N)
        frequency = np.random.uniform(*ranges_dict["frequency"], N)
        damping = np.random.uniform(*ranges_dict["damping"], N)
        speed = np.random.uniform(*ranges_dict["speed"], N)

        # Draw signal-to-noise ratio if specified.
        snr = (
            np.random.uniform(*ranges_dict["snr"], N)
            if "snr" in ranges_dict.keys()
            else None
        )

        # Draw number of series and number of spectral points per series.
        s = np.random.randint(*ranges_dict["s"])  # [0], ranges_dict["s"][1])
        n = np.random.randint(*ranges_dict["n"])  # [0], ranges_dict["n"][1])

        # Draw the length and step of Radon dimension.
        dwmin = np.random.uniform(*ranges_dict["dwmin"])
        dwmax = np.random.uniform(*ranges_dict["dwmax"])
        ddw = np.random.uniform(*ranges_dict["ddw"])

        # Create a Radon instance.
        radon = Radon(
            frequency,
            amplitude,
            damping,
            speed,
            s,
            n,
            dwmin,
            dwmax,
            ddw,
            snr,
        )

        return radon


## test


print(radon.radon_spectra)
