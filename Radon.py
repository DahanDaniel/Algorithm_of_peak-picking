from typing import Union, List, Tuple, TypeVar

import numpy as np
import matplotlib.pyplot as plt

Num = Union[int, float]
R = TypeVar("R", bound="Radon")


class Radon:
    frequency: List[Num or tuple]  # singlets or multiplets
    amplitude: List[Num]
    damping: List[Num]
    speed: List[Num]

    s: int  # number of series
    n: int  # number of spectral points
    snr: Num or None

    dwmin: Num  # Lower limit of the speed dimension
    dwmax: Num  # Upper limit of the speed dimension
    ndw: int  # Resolution of the speed dimension

    fid: np.ndarray
    spectra: np.ndarray
    radon_spectrum: np.ndarray

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
        ndw,
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
        self.ndw = ndw

        self.transform()

    def transform(self):
        # To avoid aliasing, the starting resolution is doubled and all
        # frequencies moved to the middle of the spectrum, essentialy
        # creating margins from both sides of the spectrum. All
        # frequencies out of the desired range are then muted, and the
        # resulting, doubled in size spectrum is finally trimmed to
        # reflect the original parameters.

        # Create FIDs with doubled resolution and shifted frequencies.
        n2 = 2 * self.n
        freq_shift = int(self.n / 2)
        shifted_freq = [freq + freq_shift for freq in self.frequency]
        fid = np.zeros((self.s, n2), dtype="complex64")
        t = np.linspace(0, 1, n2, endpoint=False)
        for i in range(self.s):
            for k in range(np.shape(self.amplitude)[0]):
                tot_freq = shifted_freq[k] + i * self.speed[k]
                if (
                    0.8 * freq_shift <= tot_freq <= n2 - 0.8 * freq_shift
                ):  # Mute frequencies out of the desired range.
                    exponent = (
                        2 * np.pi * 1j * (tot_freq + 1j * self.damping[k]) * t
                    )
                    fid[i] = np.add(
                        fid[i],
                        self.amplitude[k] * np.e ** exponent,
                    )
                if self.snr:  # Add random noise, skip first point.
                    noise = (
                        (1 / self.snr)
                        * np.max(self.amplitude)
                        * np.random.uniform(0, 1, n2 - 1)
                    )
                    fid[i][1:n2] = np.add(fid[i][1:n2], noise)
                    fid[i] = np.add(fid[i], -1 * np.average(noise))

        # Fix the first point for the Fast Fourier Transform.
        for i in range(self.s):
            fid[i][0] /= 2

        # Phase correction
        dw_arr = np.linspace(
            self.dwmin, self.dwmax, self.ndw
        )  # Domain of rates of change.
        p = np.zeros((len(dw_arr), *np.shape(fid)), dtype="complex64")
        a = 2 * np.pi * 1j * np.linspace(0, 1, n2, endpoint=False)
        for i in range(len(dw_arr)):
            b = a * dw_arr[i]
            for k in range(self.s):
                p[i][k] = fid[k] * np.e ** (-b * k)

        # "Diagonal" summation.
        pr = np.zeros((len(dw_arr), n2), dtype="complex64")
        for i in range(len(dw_arr)):
            for j in range(self.s):
                pr[i] += p[i][j]

        # Set fields values taking into account initial resolution doubling.
        self.fid = fid[: self.n]
        self.spectra = np.fft.fft(self.fid)[:, freq_shift : n2 - freq_shift]
        self.radon_spectrum = np.fft.fft(pr)[:, freq_shift : n2 - freq_shift]

    def generate_random_data(ranges_dict: dict) -> R:
        """
        The ranges_dict variable should contain the following Radon
        class fields' names strings as keys:
            ["frequency", "amplitude", "damping", "speed",
            "s", "n", "dwmin", "dwmax", "ndw", "snr"]
        and corresponding tuples specifying their ranges as values, as
        well as a "number_of_peaks" key with an analogous tuple of ints.
        If the "frequency" tuple contains more than two values, than the
        third value determines the maximal number of components in a
        single multiplet, and the fourth value determines the maximal
        distance between components (the amplitude parameter determines
        then the amplitude of the highest peak in the multiplet).

        If you want to set a specific value to constant just pass the
        range tuple with lower_bound == upper_bound, or
        lower_bound == upper_bound - 1 in case of "s", "n" and
        "number_of_peaks".

        General key-value pair form:
        "field_name': (lower_bound, upper_bound)

        To generate multiples in spectra the "frequency" field should be
        structured as: (lower_bound, upper_bound, max_multiplet, radius)

        If snr is not specified or set to 0, then no noise is added.
        If max_multiplets is not specified, only singlets are generated.
        """
        # Draw a random number of peaks.
        N = np.random.randint(*ranges_dict["number_of_peaks"])

        # Randomize peaks' parameters.
        damping = np.random.uniform(*ranges_dict["damping"], N)
        speed = np.random.uniform(*ranges_dict["speed"], N)
        if len(ranges_dict["frequency"]) == 4:
            frequency, amplitude = generate_multiplets(
                N,
                ranges_dict["frequency"][:2],
                ranges_dict["amplitude"],
                ranges_dict["frequency"][2],
                ranges_dict["frequency"][3],
            )
        else:
            # Only singlets
            frequency = np.random.uniform(*ranges_dict["frequency"], N)
            amplitude = np.random.uniform(
                *ranges_dict["amplitude"], len(frequency)
            )

        # Draw signal-to-noise ratio if specified.
        snr = (
            np.random.uniform(*ranges_dict["snr"])
            if "snr" in ranges_dict.keys()
            else None
        )

        # Draw number of series and number of spectral points per series.
        s = np.random.randint(*ranges_dict["s"])  # [0], ranges_dict["s"][1])
        n = np.random.randint(*ranges_dict["n"])  # [0], ranges_dict["n"][1])

        # Draw the length and step of Radon dimension.
        dwmin = np.random.uniform(*ranges_dict["dwmin"])
        dwmax = np.random.uniform(*ranges_dict["dwmax"])
        ndw = np.random.randint(*ranges_dict["ndw"])

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
            ndw,
            snr,
        )

        return radon


def get_pascal_row(n: int) -> List[int]:
    """Returns n-th row of Pascal's trangle as a list."""

    if n == 0:
        return [1]
    if n == 1:
        return [1, 1]
    row = [1, 1]
    for i in range(2, n + 1):
        prev_row = row
        row = [1]
        for i in range(len(prev_row) - 1):
            row.append(prev_row[i] + prev_row[i + 1])
        row.append(1)

    return row


def total_int_into_rand_list(total: int, max_el: int) -> List[int]:
    """This function creates a random list of non-zero integers that sum
    up to the given number, ie. f(7, 3) -> [2, 1, 3, 1]"""

    result_list = []
    sum = 0
    while sum != total:
        rand_num = np.random.randint(1, max_el + 1)
        if sum + rand_num < total:
            result_list.append(rand_num)
            sum += rand_num
        else:
            num = total - sum
            result_list.append(num)
            sum += num  # break

    return result_list


def generate_multiplets(
    n_o_peaks: int,
    freq_bounds: tuple,
    amp_bounds: tuple,
    max_multiplet: int,
    max_radius: float,
) -> Tuple[List[Num], List[Num]]:
    multiplets_list = total_int_into_rand_list(n_o_peaks, max_multiplet)

    frequency = []
    amplitude = []
    for multiplet_size in multiplets_list:
        # Generate frequencies averaging to the drawn frequency, radius
        # apart from each other.
        radius = np.random.uniform(max_radius)
        main_freq = np.random.uniform(*freq_bounds)
        freqs = np.linspace(
            main_freq - multiplet_size * radius,
            main_freq + multiplet_size * radius,
            multiplet_size,
            endpoint=True,
        )
        frequency = np.concatenate((frequency, freqs)).tolist()

        # Generate amplitudes with proper proportions.
        main_amp = np.random.uniform(*amp_bounds)
        pascal_row = get_pascal_row(multiplet_size - 1)
        scaling_list = [n / max(pascal_row) for n in pascal_row]
        amps = [scale_factor * main_amp for scale_factor in scaling_list]
        amplitude = np.concatenate((amplitude, amps)).tolist()

    return frequency, amplitude


if __name__ == "__main__":
    for _ in range(2):
        ranges_dict = {
            "number_of_peaks": (1, 10),
            "frequency": (
                0,
                256,
                4,
                10,
            ),  # lower_bound, upper_bound, max_multiplet, max_radius
            "amplitude": (1, 5),
            "damping": (0.1, 10),
            "speed": (-10, 10),
            "s": (20, 21),
            "n": (256, 257),
            "dwmax": (10, 10),
            "dwmin": (-10, -10),
            "ndw": (256, 257),
            "snr": (10, 100),
        }
        radon = Radon.generate_random_data(ranges_dict)

        fig = plt.figure()
        plt.plot(radon.spectra[0].real)
        plt.show(block=False)

        fig = plt.figure()
        plt.imshow(radon.radon_spectrum.real.T)
        plt.show(block=False)

    plt.show()
