import time
import numpy as np
from multiprocessing import cpu_count
from numba import njit, prange #set_num_threads, get_num_threads)
from ghostipy.spectral.cwt import cwt
from ghostipy.spectral.wavelets import (MorseWavelet, AmorWavelet, BumpWavelet)

__all__ = ['wsst']

@njit(parallel=True)
def assign_to_frequency_bins(W, dW, freq_vec, voices_per_octave, eps, fs, n_threads):
    #n_threads_original = get_num_threads() # only available on 0.49.1? fix
    #set_num_threads(n_threads)
    J, T = W.shape
    sst = np.zeros((freq_vec.shape[0], W.shape[1]), dtype=W.dtype)
    
    #phasetf = (dW / W).imag / (2 * np.pi)
    #freq_inds = np.zeros(W.shape[1], dtype=np.int32)
    phasetf = np.zeros(T)
    k = np.zeros(T, dtype=np.int64)
    
    for j in range(J):
        for m in prange(T):
            if np.abs(W[j, m]) > eps:
                phasetf[m] = (dW[j, m] / W[j, m]).imag / (2 * np.pi) * fs
                if (np.isfinite(phasetf[m])
                and phasetf[m] <= freq_vec[0]
                and phasetf[m] >= freq_vec[-1]):
                    k[m] = np.argmin(np.abs(freq_vec - phasetf[m]))
                    #print(j, m, phasetf, k)
                    sst[k[m], m] += W[j, m] * np.log(2) / voices_per_octave
          
#             freq_ind = J - np.floor(J/(log2f2 - log2f1) * (np.log2(phasetf) - log2f1))
#             if (np.isfinite(freq_ind) == True
#             and freq_ind > -1
#             and freq_ind < J
#             and np.abs(W[j, m]) > eps):
                
#                 sst[int(freq_ind), m] = sst[int(freq_ind), m] + W[j, m] * np.log(2) / voices_per_octave
                
    # only available on 0.49.1? fix
    # set_num_threads(n_threads_original)
    return sst

@njit(cache=True)
def assign_to_frequency_bins_serial(W, dW, freq_vec, voices_per_octave, eps, fs):
    J, T = W.shape
    sst = np.zeros((freq_vec.shape[0], W.shape[1]), dtype=W.dtype)
    
    #phasetf = (dW / W).imag / (2 * np.pi)
    #freq_inds = np.zeros(W.shape[1], dtype=np.int32)
    
    for j in range(J):
        for m in range(T):
            if np.abs(W[j, m]) > eps:
                phasetf = (dW[j, m] / W[j, m]).imag / (2 * np.pi) * fs
                if (np.isfinite(phasetf)
                    and phasetf <= freq_vec[0]
                    and phasetf >= freq_vec[-1]):

                    k = np.argmin(np.abs(freq_vec - phasetf))
                    #print(j, m, phasetf, k)
                    sst[k, m] += W[j, m] * np.log(2) / voices_per_octave
          
#             freq_ind = J - np.floor(J/(log2f2 - log2f1) * (np.log2(phasetf) - log2f1))
#             if (np.isfinite(freq_ind) == True
#             and freq_ind > -1
#             and freq_ind < J
#             and np.abs(W[j, m]) > eps):
                
#                 sst[int(freq_ind), m] = sst[int(freq_ind), m] + W[j, m] * np.log(2) / voices_per_octave
                
    return sst

def compute_sst_chunk(cwt_computed, dcwt_computed, freqs, voices_per_octave,
                      fs, start, n_time_points, eps, sst_out):

    cwt_chunk = cwt_computed[:, start:start+n_time_points]
    dcwt_chunk = dcwt_computed[:, start:start+n_time_points]
    sst_out[:, start:start + n_time_points] = assign_to_frequency_bins_serial(
        cwt_chunk,
        dcwt_chunk,
        freqs,
        voices_per_octave,
        eps,
        fs)

def wsst(data, *, eps=1e-8, fs=1, timestamps=None, wavelet=MorseWavelet(gamma=3, beta=20),
         freq_limits=None, voices_per_octave=32, n_workers=cpu_count(),
         verbose=False, method='full', remove_mean=False, boundary='mirror',
         coi_threshold=1/(np.e**2), describe_dims=False, cwt_out=None,
         sst_out=None, write_chunk_size='1G'):

    """Computes the wavelet synchrosqueezed transform

    Parameters
    ----------
    data : np.ndarray, with shape (n_timepoints, )
        Data with which to compute the CWT
    eps : float, optional
        Minimum absolute value of the CWT coefficients that are
        considered nonzero.
        Default is 1e-8.
    fs : float
        Sampling rate of the data in Hz.
    timestamps : np.ndarray, with shape (n_timepoints, ) optional
        Timestamps corresponding to the data, in seconds.
        If None, they will be computed automatically based on the
        assumption that all the data are one contiguous block.
    wavelet : ghostipy.wavelet
        Type of wavelet to use for the transform.
        Default is a Morse wavelet with beta=3 and gamma=20.
    freq_limits : list, optional
        List of [lower_bound, upper_bound] for frequencies to use,
        in units of Hz. Note that a reference set of frequencies
        is generated on the shortest segment of data since that
        determines the lowest frequency that can be used. If the
        bounds specified by 'freq_limits' are outside the bounds
        determined by the reference set, 'freq_limits' will be
        adjusted to be within the bounds of the reference set.
    voices_per_octave : int, optional
        Number of wavelet frequencies per octave.
        Default is 32.
    n_workers : integer, optional
        Number of parallel jobs to use.
        Default is the total number of CPUs (which may be virtual).
    verbose : boolean, optional
        Whether to print messages displaying this function's progress.
        Default is False.
    method: {'full', 'ola'}, optional
        Which algorithm to use for computing the CWT. 'ola' will give
        superior performance for long lengths of data.
    derivative: boolean, optional
        Whether to compute the derivative of the CWT.
        Default is False.
    remove_mean: boolean, optional
        Whether to remove the mean of the data before computing the CWT.
        Default is False.
    boundary: {'mirror', 'zeros', 'periodic'}, optional
        To handle boundaries, the data are extended before computing the CWT
        according to the following options:
        'mirror' : Mirror/reflect the data at each end
        'zeros': Add zeros at each end
        'periodic': Treat the data as periodic
        Note that regardless of the boundary method, the CWT should not be
        treated as reliable within the cone of influence.
        Default is 'mirror'.
    coi_threshold : float, optional
        The value C that determines the wavelet's cone of influence. The
        maximum value P of the wavelet's power autocorrelation is taken
        in the time domain. Then the cone of influence is given by the
        region where the power autocorrelation is above C*P. Default value
        for C is e^(-2).
    describe_dims : boolean, optional
        Whether to return the expected shape and dtype of the intermediate
        and final output arrays and return immediately (no SST is computed).
        This option is useful for out-of-core computation. While the expected
        shape should not be changed, the dtype is only suggested, e.g. it is
        acceptable to use a lower precision dtype (i.e. complex64 instead
        of complex128 to save space).
        Default is False.
    cwt_out: array-like, optional
        If specified, the CWT coefficients (used in an intermediate step)
        will be stored here. Useful if there is not enough space in memory
        must instead be saved to an array stored on disk.
    sst_out : array-like, optional
        If specified, the final wavelet SST coefficients will be stored here.
        Useful if the output is too large to fit into memory and must instead
        be saved to an array stored on disk.
    write_chunk_size : string, optional
        Maximum chunk size to use when writing to the final SST output
        array. This option is most useful if the array is stored on disk
        (specifically, too many write calls may result in lower performance).
        Use 'K' for kilobytes, 'M' for megabytes, and 'G' for gigabytes.
        Default is 1 GB.

    Returns
    -------
    If 'describe_dims' is True:
        cwt_shape, cwt_dtype, sst_shape, sst_dtype : tuple
            A tuple consisting of:
            (1) expected shape of the intermediate CWT coefficient array,
            (2) dtype of the intermediate CWT array,
            (3) expected shape of the final output SST coefficient array,
            (4) dtype of the final output SST coefficient array.
    Otherwise:
        coefs : np.ndarray, with shape (n_scales, n_timepoints)
            Calculated continuous wavelet coefficients. Note that the scale
            dimension is ordered by increasing wavelet scale, which corresponds
            to decreasing frequency.
        scales : np.ndarray, with shape (n_scales, )
            Wavelet scales for which CWT was calculated.
        frequencies : np.ndarray, with shape (n_frequencies, )
            If a sampling rate is given, these will be the frequency
            corresponding to each wavelet scale.
        timestamps : np.array, with shape (n_timepoints, )
            If timestamps were not specified, these are the timestamps
            calculated for data samples. Pass through of input 
            timestamps otherwise.
        cois : np.ndarray, with shape (n_cois, )
            Cones of influence for each wavelet scale.
    """

    t0 = time.time()
    
    if describe_dims:
        if verbose:
            print("Calculating output array sizes. Skipping SST transform")
        shape, dtype = cwt(
            data, fs=fs,
            timestamps=timestamps,
            wavelet=wavelet,
            freq_limits=freq_limits, 
            voices_per_octave=voices_per_octave,
            n_workers=n_workers,
            verbose=verbose,
            method=method,
            remove_mean=remove_mean,
            boundary=boundary,
            coi_threshold=coi_threshold,
            describe_dims=describe_dims)

        # same shape and dtype for CWT and SST
        return shape, dtype, shape, dtype

    if verbose:
        print("Computing CWT")
    cwt_computed, scales, freqs, timestamps, cois = cwt(
            data, fs=fs,
            timestamps=timestamps,
            wavelet=wavelet,
            freq_limits=freq_limits, 
            voices_per_octave=voices_per_octave,
            n_workers=n_workers,
            verbose=verbose,
            method=method,
            derivative=False,
            remove_mean=remove_mean,
            boundary=boundary,
            coi_threshold=coi_threshold,
            describe_dims=describe_dims,
            cwt_out=cwt_out)

    if verbose:
        print("Computing the CWT derivative")
    # fill output array with derivative, will be overwritten later with SST data
    dcwt_computed, _, _, _, _ = cwt(
            data, fs=fs,
            timestamps=timestamps,
            wavelet=wavelet,
            freq_limits=freq_limits, 
            voices_per_octave=voices_per_octave,
            n_workers=n_workers,
            verbose=verbose,
            method=method,
            derivative=True,
            remove_mean=remove_mean,
            boundary=boundary,
            coi_threshold=coi_threshold,
            describe_dims=describe_dims,
            cwt_out=sst_out)
    
    # compute sst on chunks
    A, T = dcwt_computed.shape
    n_tot_elements = dcwt_computed.size
    n_bytes_per_element = dcwt_computed.dtype.itemsize

    if write_chunk_size is None:
        n_bytes = n_tot_elements * n_bytes_per_element
    elif write_chunk_size[-1] == 'K':
        n_bytes = int(write_chunk_size[:-1]) * 1000
    elif write_chunk_size[-1] == 'M':
        n_bytes = int(write_chunk_size[:-1]) * 1000**2
    elif write_chunk_size[-1] == 'G':
        n_bytes = int(write_chunk_size[:-1]) * 1000**3
    else:
        raise ValueError("Unsupported value of write_chunk_size")
    
    if n_bytes > n_tot_elements * n_bytes_per_element:
        n_time_points = T
    else:
        n_time_points = int(np.floor(n_bytes / (A * n_bytes_per_element)))
    starts = np.arange(0, T, n_time_points)

    sst_out = dcwt_computed
#     dcwt_computed_2 = copy.deepcopy(dcwt_computed)
#     sst_out_2 = dcwt_computed_2
    
#     t0 = time.time()
    for ii, start in enumerate(starts):
        if verbose:
            print(f"Computing SST chunk {ii} of {len(starts) - 1}")
        
        cwt_chunk = cwt_computed[:, start:start+n_time_points]
        dcwt_chunk = dcwt_computed[:, start:start+n_time_points]
        n_actual_points = cwt_chunk.shape[1]
        sst_out[:, start:start + n_actual_points] = assign_to_frequency_bins(
            cwt_chunk,
            dcwt_chunk,
            freqs,
            voices_per_octave,
            eps,
            fs,
            n_workers)

    if verbose:
        print(f'WSST total elapsed time: {time.time() - t0} seconds')
        
#     print(f"Finished in {time.time() - t0} seconds")
    
#     t0 = time.time()
#     for ii, start in enumerate(starts):
#         print(f"Computing SST chunk {ii} of {len(starts) - 1}")
#         cwt_chunk = cwt_computed[:, start:start+n_time_points]
#         dcwt_chunk = dcwt_computed_2[:, start:start+n_time_points]
#         sst_out_2[:, start:start + n_time_points] = assign_to_frequency_bins_serial(cwt_chunk,
#                                                                            dcwt_chunk,
#                                                                            freqs,
#                                                                            voices_per_octave,
#                                                                            eps,
#                                                                            fs)
        
#     print(f"Finished in {time.time() - t0} seconds")

#     print("verifying correctness")
#     assert np.allclose(sst_out, sst_out_2)

    # task_list = []
    # for ii, start in enumerate(starts):
    #     task = dask.delayed(compute_sst_chunk)(cwt_computed,
    #                                            dcwt_computed,
    #                                            freqs,
    #                                            voices_per_octave,
    #                                            fs,
    #                                            start,
    #                                            n_time_points,
    #                                            eps,
    #                                            sst_out)
    #     task_list.append(task)
        
    # with ProgressBar():
    #     dask.compute(task_list, n_workers=n_workers)
        
    return sst_out, scales, freqs, timestamps, cois


###################################################################
# old stuff
# @njit(cache=True)
# def assign_values_nb(cwt, freq_inds, a_inds, t_inds, nvoices, scales):

#     sst = np.zeros_like(cwt)

#     for i in range(a_inds.shape[0]):
        
#         j = a_inds[i]
#         m = t_inds[i]
# #         ax_ind_0 = ax_inds_0[i]
# #         ax_ind_1 = ax_inds_1[i]
        
#         k = int(freq_inds[j, m])
#         sst[k, m] = sst[k, m] + cwt[j, m] / nvoices * np.log(2)#/np.sqrt(scales[ax_ind_0])

#     return sst

# @njit(parallel=True, cache=True)
# def assign_to_frequency_bins(W, phasetf, log2f1, log2f2, voices_per_octave, eps):
#     J, T = W.shape
#     sst = np.zeros_like(W)
#     freq_inds = np.zeros(W.shape[1], dtype=np.int32)
#     for j in range(J):
#         for m in prange(T):
#             freq_inds[m] = J - np.floor(J/(log2f2 - log2f1) * (np.log2(phasetf[j, m]) - log2f1))
#             if (np.isnan(freq_inds[m]) == False \
#                 and freq_inds[m] > -1
#                 and freq_inds[m] < J 
#                 and np.abs(W[j, m]) > eps):
                
#                 sst[freq_inds[m], m] = sst[freq_inds[m], m] + W[j, m] * np.log(2) / voices_per_octave
                
#     return sst

# @njit(parallel=True)
# def assign_to_frequency_bins(W, dW, log2f1, log2f2, voices_per_octave, eps, fs):
#     phasetf = (dW/W).imag / (2*np.pi) * fs
#     T = W.shape[0]
#     sst = np.zeros(T, dtype='<c16')
#     for m in prange(T):
#         freq_inds[m] = J - np.floor(J/(log2f2 - log2f1) * (np.log2(phasetf[j, m]) - log2f1))
#         if (np.isnan(freq_inds[m]) == False \
#             and freq_inds[m] > -1
#             and freq_inds[m] < J 
#             and np.abs(W[j, m]) > eps):

#             sst[freq_inds[m], m] = sst[freq_inds[m], m] + W[j, m] * np.log(2) / voices_per_octave
                
#     return sst


# def sst(data, *, eps=1e-8, timestamps=None, fs=1, wavelet=BumpWavelet(),
#         freq_limits=None, voices_per_octave=32, out_of_core=False,
#         boundary='periodic', remove_mean=False, threads=cpu_count(), coi_threshold=1/(np.e**2)):
    
    
#     N = data.shape[0]

#     f_ref_low = fs / N
#     f_ref_high = fs / 2

# #     if freqs is not None:
# #         # just in case user didn't pass in sorted
# #         # frequencies after all
# #         freqs = np.sort(freqs)
# #         # freq_bounds = [freqs[0], freqs[-1]]
# #         # lb, ub = self._check_freq_bounds(freq_bounds, freq_bounds_ref)
# #         # mask = np.logical_and(freqs >= lb, freqs <= ub)
# #         # f = freqs[mask]
# #         ws = hz_to_normalized_rad(freqs, fs)
# #         w_low = ws[0]
# #         w_high = ws[-1]
# #         if w_low < w_ref_low:
# #             logging.warning("Lower frequency limit of {} is less than the smallest allowed"
# #                   " frequency of {} Hz".format(freq_limits[0], normalized_rad_to_hz(w_ref_low)))
# #         if w_high > w_ref_high:
# #            logging.warning("Upper frequency limit of {} is greater than the largest allowed"
# #                   " frequency of {} Hz".format(freq_limits[1], normalized_rad_to_hz(w_ref_high)))
#     if freq_limits is not None:
#         # just in case user didn't pass in limits as
#         # [lower_bound, upper_bound]
#         # freq_limits = np.sort(freq_limits)
#         # freq_bounds = [freq_limits[0], freq_limits[1]]
#         # f_low, f_high = self._check_freq_bounds(freq_bounds, freq_bounds_ref)
#         f_low = freq_limits[0]
#         f_high = freq_limits[-1]
#         if f_low < f_ref_low:
#             logging.warning("Lower frequency limit of {} is less than the smallest allowed"
#                   " frequency of {} Hz".format(freq_limits[0], f_ref_low))
#         if f_high > f_ref_high:
#             logging.warning("Upper frequency limit of {} is greater than the largest allowed"
#                   " frequency of {} Hz".format(freq_limits[1], f_ref_high))
#     else:
#         f_low = f_ref_low
#         f_high = f_ref_high


# #     n_octaves = np.log2(w_high / w_low)
# #     j = np.arange(n_octaves * voices_per_octave)
# #     ws = w_high / 2**(j/voices_per_octave)

# #     scales = wavelet.freq_to_scale(ws)
# #     cois = wavelet.coi(scales, ref_scale, ref_coi)
    
    
    
# #     N = len(data)
# #     dt = 1/1000
    
# #     f1 = 1 / N * 2 * np.pi  # normalized radian frequency
# # #     f1 = 5 / 1250 * 2 * np.pi
# #     f2 = np.pi
# # #     f2 = 500 / 1250 * 2 * np.pi
# #     eps = 1e-8

# #     n_octaves = np.log2(f2 / f1)
# #     j = np.arange(n_octaves * voices_per_octave)
# #     freqs = f1 * 2**(j/voices_per_octave)
# #     scales = wavelet.freq_to_scale(freqs)

#     if out_of_core:
#         cwt_filename = '/home/jchu/Downloads/cwt.hdf5'
#         dcwt_filename = '/home/jchu/Downloads/dcwt.hdf5'
#     else:
#         cwt_filename = None
#         dcwt_filename = None

#     W, scales, freqs, coi_w = cwt(data, timestamps=timestamps, fs=fs, wavelet=wavelet,
#                  freq_limits=[f_low, f_high], voices_per_octave=voices_per_octave,
#                  out_of_core_filename=cwt_filename, 
#                  boundary=boundary, remove_mean=remove_mean,
#                  threads=threads, coi_threshold=coi_threshold)
#     dW, scales2, freqs2, coi_dw = cwt(data, timestamps=timestamps, fs=fs, wavelet=wavelet,
#                  freq_limits=[f_low, f_high], voices_per_octave=voices_per_octave, 
#                  out_of_core_filename=cwt_filename, 
#                  boundary=boundary, remove_mean=remove_mean,
#                  threads=threads, coi_threshold=coi_threshold,
#                  derivative=True)
    
#     assert np.allclose(freqs, freqs2)
#     assert np.allclose(scales, scales2)

# #     W = np.zeros((len(scales), N), dtype='<c16')
# #     dW = np.zeros_like(W)
    
# #     #data2 = data - np.mean(data)
# #     data_fft = np.fft.fft(data)
    
# #     psif, dpsif = wavelet(N, scales, return_derivative=True)
    
# #     assert psif.shape == (M, N)
# #     assert dpsif.shape == psif.shape
    
# #     W = np.fft.ifft(data_fft * psif)
# #     dW = np.fft.ifft(data_fft * dpsif)
    
#     phasetf = (dW / W).imag / (2 * np.pi) * fs
    
#     log2f1 = np.log2(f_low)  # normalized frequency with fs=1 Hz
#     log2f2 = np.log2(f_high)  # normalized frequency with fs=1 Hz
# #     del dW
# #     gc.collect()

# #     na = len(scales)
# #     dw = 1 / ((na - 1) * np.log2(N/2))
# #     log2f1 = np.log2(5 / 1250)
# #     log2f2 = np.log2(625 / 1250)
    
#     t0 = time.time()
#     M = W.shape[0]
#     with np.errstate(invalid='ignore'):
#         #freq_inds = np.minimum( np.maximum(np.round(1/dw * (np.log2(phasetf) - log2f1)), 0), na-1)
#         freq_inds = M - np.floor(M/(log2f2 - log2f1) * (np.log2(phasetf) - log2f1))
#         masked_freq_inds = np.ma.array(freq_inds, mask=np.isnan(freq_inds))
#         #phasetf_mask = np.logical_and(masked_freq_inds > -1, masked_freq_inds < M)
#         phasetf_mask = np.logical_and(~np.isnan(freq_inds), np.logical_and(freq_inds > -1, freq_inds < M))
#     cwt_mask = np.abs(W) > eps

#     inds = np.where(phasetf_mask & cwt_mask)

#     Tx = assign_values_nb(W, freq_inds, inds[0], inds[1], voices_per_octave, scales)
#     print("Elapsed time 1: {} seconds".format(time.time() - t0))

#     t0 = time.time()
#     Tx2 = assign_to_frequency_bins(W, phasetf, log2f1, log2f2, voices_per_octave, eps)
#     print("Elapsed time 2: {} seconds".format(time.time() - t0))
    
#     assert np.allclose(Tx, Tx2)

#     return freqs, W, dW, Tx, Tx2

# def reconstruct(Tx, freqs, freq_limits, wavelet):
    
#     mask = np.logical_and(freqs>freq_limits[0], freqs<freq_limits[-1])
    
#     c_psi = wavelet.admissibility_constant
    
#     xrec = 1/c_psi * 2 * np.real(np.sum(Tx[mask, :], axis=0))
    
#     return xrec

# # To compute power spectrum:
# # 1. Fourier: Take rfft. Divide 0 Hz component by 2 (and Nyquist by 2 if signal is even). Then square fft components
# # 2. Wavelet: Square wavelet transform, sum across time, multiply by length of signal and divide by 2
# # 3. SST: Divide by admissibility constant, square and sum across time, multiply by length of signal and multiply by 2

# # To compute PSD
# # 1. Fourier: Compute power spectrum as above but also divide by N and by fs
# # 2. Wavelet: Square wavelet transform, sum across time, divide by fs and divide by 2
# # 3. SST : Divide by admissibility consant, square and sum across time, divide by fs and multiply by 2
# # 4. Hilbert: Compute analytic signal, square and sum across time, divide by fs and divide by 2




# # Algorithm for SST (including out of core support)
# # 1. Chunk wavelet into blocks and use dask delayed over blocks
# # 2. For each block:
# # 3. Compute and return SST
# # 4. Override CWT with SST value (only need to allocate one array)