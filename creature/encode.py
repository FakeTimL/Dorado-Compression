import convolutional
# from simulation import simulate_errors
# from analysis import get_errors

import os
import numpy as np
import time


def ascii_to_binary(f='encoding_data/data.txt', b='encoding_data/data_bin.txt', max_len=None):
    """
        Converts ASCII file to binary text file.
        If max_len is set, splits the output binary into sequences of length max_len.
    """
    with open(f, 'r') as txtfile:
        text = txtfile.read()

    bin_seq = ''.join('{0:08b}'.format(ch, 'b') for ch in bytearray(text, 'utf-8'))
    if max_len:
        bin_seq = '\n'.join([bin_seq[0+i:max_len+i] for i in range(0, len(bin_seq), max_len)])

    with open(b, 'w') as binfile:
        binfile.write(bin_seq)

    return bin_seq


def binary_to_ascii(b='encoding_data/data_bin.txt', output='encoding_data/output.txt'):
    """
        Converts binary text file to ASCII file.
    """
    with open(b, 'r') as binfile:
        bin_seq = binfile.read()
    bin_seq = bin_seq.replace('\n', '')

    with open(output, 'w') as out:
        # Assuming utf-8
        decoded = ''.join(chr(int(bin_seq[i*8:i*8+8],2)) for i in range(len(bin_seq)//8))
        out.write(decoded)

    return decoded


def binary_to_nt(b='encoding_data/data_bin.txt', output='encoding_data/data_nuc.txt'):
    """
        Converts binary text file to file consisting of data encoded into DNA nucleotides.
    """
    with open(b, 'r') as binfile:
        bin_seqs = binfile.readlines()

    dna_seqs = [binary_to_bases(bin_seq.strip()) for bin_seq in bin_seqs]

    with open(output, 'w') as out:
        out.write('\n'.join(dna_seqs))

    return dna_seqs


def nt_to_binary(nt='encoding_data/data_nuc.txt', output='encoding_data/data_bin.txt'):
    """
        Converts file consisting of data encoded into DNA nucleotides into binary text file.
    """
    with open(nt, 'r') as ntfile:
        dna_seqs = ntfile.readlines()

    bin_seqs = [bases_to_binary(dna_seq.strip()) for dna_seq in dna_seqs]

    with open(output, 'w') as out:
        out.write('\n'.join(bin_seqs))

    return bin_seqs


def binary_to_bases(bin_seq):
    """
        Converts binary string sequence to DNA nucleotide sequence.
    """
    mapping = {'00': 'A', '01': 'C', '10': 'G', '11': 'T'}
    return ''.join(mapping[bin_seq[i*2:i*2+2]] for i in range(len(bin_seq)//2))


def bases_to_binary(dna_seq):
    """
        Converts DNA nucleotide sequence to binary string sequence.
    """
    mapping = {'A': '00', 'C': '01', 'G': '10', 'T': '11'}
    return ''.join(mapping[nt] for nt in dna_seq)


def encode(fsm, f='encoding_data/data_test.txt', bin_max_len=200):
    """
        Encodes ASCII text file using FSM-based code into DNA nucleotide sequences.
    """
    filename = os.path.splitext(f)[0]
    binfile = filename+'_bin.txt'
    print("Converting to binary...")
    bin_seqs = ascii_to_binary(f, binfile, max_len=bin_max_len)
    print("Conversion to binary complete.")

    binary_to_nt(b=binfile, output=filename+'_dna.txt') # For analysis later

    bin_streams = [[int(bin) for bin in bin_seq] for bin_seq in bin_seqs.split('\n')]

    print("Encoding...")
    start = time.process_time()
    encoded_streams = []
    for bin_stream in bin_streams:
        fsm.presentState = 0 # TODO: make a function in FSM to reset
        fsm.presentStateCoordinate = 0
        encoded_streams.append(convolutional.FSMEncoder(bin_stream, fsm))
    print("Encoding complete in %s seconds." % (time.process_time() - start))

    encoded_flatstreams = []
    for encoded_stream in encoded_streams:
        encoded_flatstream = []
        for sublist in encoded_stream:
            for item in sublist:
                encoded_flatstream.append(str(item))
        encoded_flatstreams.append(''.join(encoded_flatstream))

    encoded_binfile = filename+'_encoded_bin.txt'
    with open(encoded_binfile, 'w') as enc:
        for encoded_flatstream in encoded_flatstreams:
            enc.write("%s\n" % encoded_flatstream)

    encoded_dnafile = filename+'_encoded_dna.txt'
    return binary_to_nt(b=encoded_binfile, output=encoded_dnafile)


def decode(fsm, f='encoding_data/data_test_encoded_dna.txt', out_root='encoding_data/data_test', symbol_size=3):
    """
        Decodes DNA nucleotide sequences using Viterbi decoder and FSM into decoded ASCII file.
    """
    read_encoded_bins = nt_to_binary(nt=f, output=out_root+'_read_encoded_bin.txt')

    def myFanOutFunction(state, observation, time):
        return convolutional.genericFanOutFunction(fsm, state, observation, time, None)

    decoded_streams = []
    start = time.process_time()
    for i, read_encoded_bin in enumerate(read_encoded_bins):
        print("Decoding %s of %s..." % (i, len(read_encoded_bins)))
        read_encoded_bin = [int(bin) for bin in read_encoded_bin]
        decoded_streams.append(convolutional.viterbiDecoderWithFlagging(8, 0, myFanOutFunction, read_encoded_bin, symbol_size, produceGraphics=False)[0][0].pathTriggers)
    print("Decoding complete in %s seconds." % (time.process_time() - start))

    decoded_flatstreams = []
    for decoded_stream in decoded_streams:
        decoded_flatstream = []
        for sublist in decoded_stream:
            for item in sublist:
                decoded_flatstream.append(str(item))
        decoded_flatstreams.append(''.join(decoded_flatstream))

    decoded_bin_file = out_root+'_read_decoded_bin.txt'
    with open(decoded_bin_file, 'w') as decoded_file:
        for seq in decoded_flatstreams:
            decoded_file.write("%s\n" % ''.join(seq))

    binary_to_nt(b=decoded_bin_file, output=out_root+'_read_decoded_dna.txt') # For analysis later
    binary_to_ascii(b=decoded_bin_file, output=out_root+'_read.txt')


# def test_ecc(data='half_ecc_test_300_more_data/data.txt', rate='1/3', error_rate=None, short_oligos=False):
#     """
#         Encodes ASCII text file using error correcting code, runs DNA sequences through error simulator, then
#         decodes the simulated sequences and measures the error rate of data read back.
#         Parameters
#         ----------
#         data : string
#             ASCII text file to encode
#         rate : string, optional
#             Code rate for convolutional code to use. Must be either '1/3', '1/2' or '2/3'
#         error_rate : float, optional
#             Error rate for error simulator to use to simulate errors in DNA storage channel. If none specified,
#             uses default error rate of channel.
#         short_oligos : bool, optional
#             If true, will encode data into DNA sequences of length 150nt. Else, encodes data into DNA sequences of length 300nt.
#     """

#     print(data)
#     filename_root = os.path.splitext(data)[0]
#     print("Testing error rate of %s" % error_rate)

#     if rate == '1/3':
#     # ONE THIRD
#         print("Using 1/3 code")
#         states = [0,1,2,3,4,5,6,7]
#         triggers = [[0], [1]]
#         nextStateTable = np.array([[0,4], [0,4], [1,5], [1,5], [2,6], [2,6] , [3,7], [3,7] ])
#         outputTable = [[[0,0,0], [1,0,0]],
#                        [[0,1,1], [1,1,1]],
#                        [[0,1,0], [1,1,0]],
#                        [[0,0,1], [1,0,1]],
#                        [[0,0,1], [1,0,1]],
#                        [[0,1,0], [1,1,0]],
#                        [[0,1,1], [1,1,1]],
#                        [[0,0,0], [1,0,0]]]
#         symbolSize = 3
#         initialState = 0
#         bin_max_len = 100 if short_oligos else 200

#     elif rate == '1/2':
#     # HALF
#         print("Using 1/2 code")
#         states = [0,1,2,3]
#         triggers = [[0], [1]]
#         nextStateTable = np.array([[0,1], [2,3], [0,1], [2,3]])
#         outputTable = [[[0,0], [1,1]],
#                        [[0,1], [1,0]],
#                        [[1,1], [0,0]],
#                        [[1,0], [0,1]]]
#         symbolSize = 2
#         initialState = 0
#         bin_max_len = 150 if short_oligos else 300

#     else:
#         print("Using 2/3 code")
#         states = [0,1,2,3,4,5,6,7]
#         triggers = [[0,0], [0,1], [1,0], [1,1]]
#         nextStateTable = np.array([[0,1,2,3], [4,5,6,7], [1,0,3,2], [5,4,7,6], [2,3,0,1], [6,7,4,5] , [3,2,1,0], [7,6,5,4] ])
#         outputTable = [[[0,0,0], [1,0,0], [0,1,0], [1,1,0]],
#                        [[0,0,1], [1,0,1], [0,1,1], [1,1,1]],
#                        [[1,0,0], [0,0,0], [1,1,0], [0,1,0]],
#                        [[1,0,1], [0,0,1], [1,1,1], [0,1,1]],
#                        [[0,1,0], [1,1,0], [0,0,0], [1,0,0]],
#                        [[0,1,1], [1,1,1], [0,0,1], [1,0,1]],
#                        [[1,1,0], [0,1,0], [1,0,0], [0,0,0]],
#                        [[1,1,1], [0,1,1], [1,0,1], [0,0,1]]]
#         symbolSize = 3
#         initialState = 0
#         bin_max_len = 200 if short_oligos else 400


#     fsm = convolutional.FSM(states, triggers, outputTable, nextStateTable, initialState)

#     encoded_dna = encode(fsm, data, bin_max_len=bin_max_len)

#     print("Simulating errors...")
#     start = time.process_time()
#     read_dna = simulate_errors(encoded_dna, basic=False, error_rate=error_rate, no_indels=True, print_error_summary=True)[0]
#     filename = filename_root+'_read_encoded_dna.txt'
#     with open(filename, 'w') as read:
#         for seq in read_dna:
#             read.write("%s\n" % seq)

#     print("Error simulation complete in %s seconds." % (time.process_time() - start))

#     decode(fsm, filename, out_root=filename_root, symbol_size=symbolSize)

#     # Analysis
#     ins_rates = []
#     del_rates = []
#     sub_rates = []
#     err_rates = []

#     orig = open(filename_root+'_dna.txt')
#     read = open(filename_root+'_read_decoded_dna.txt')

#     orig_seq = orig.readline()
#     while orig_seq:
#         orig_seq = orig_seq.strip()
#         read_seq = read.readline().strip()
#         print(orig_seq)
#         print(read_seq)
#         ins, dels, subs, errs = get_errors(orig_seq, read_seq, True)
#         ins_rates.append(ins)
#         del_rates.append(dels)
#         sub_rates.append(subs)
#         err_rates.append(errs)
#         orig_seq = orig.readline()

#     orig.close()
#     read.close()

#     # Last seq is ignored as has larger number of errors if small sequence
#     print("AVGS: Ins: %s, dels: %s, subs: %s, err: %s" % (np.mean(ins_rates[:-1]), np.mean(del_rates[:-1]), np.mean(sub_rates[:-1]), np.mean(err_rates[:-1])))


# def test_no_ecc(data='no_ecc_test_150_more_data/data.txt', short_oligos=False):
#     """
#         Maps ASCII text file directly into DNA sequences with no error correcting code, runs DNA sequences through error simulator,
#         then decodes the simulated sequences and measures the error rate of data read back.
#         Parameters
#         ----------
#         data : string
#             ASCII text file to encode
#         short_oligos : bool, optional
#             If true, will encode data into DNA sequences of length 150nt. Else, encodes data into DNA sequences of length 300nt.
#     """

#     filename_root = os.path.splitext(data)[0]
#     print(data)

#     binfile = filename_root+'_bin.txt'
#     print("Converting to binary...")
#     bin_max_len = 300 if short_oligos else 600
#     bin_seqs = ascii_to_binary(data, binfile, max_len=bin_max_len)
#     print("Conversion to binary complete.")

#     encoded_dna = binary_to_nt(b=binfile, output=filename_root+'_dna.txt') # For analysis later

#     print("Simulating errors...")
#     read_dna = simulate_errors(encoded_dna, basic=False, no_indels=True)[0]
#     read_filename = filename_root+'_read_dna.txt'
#     with open(read_filename, 'w') as read:
#         for seq in read_dna:
#             read.write("%s\n" % seq)

#     print("Error simulation complete.")
#     read_bin_file = filename_root+'_read_bin.txt'
#     nt_to_binary(nt=read_filename, output=read_bin_file)

#     binary_to_ascii(b=read_bin_file, output=filename_root+'_read.txt')

#     # Analysis
#     ins_rates = []
#     del_rates = []
#     sub_rates = []
#     err_rates = []

#     orig = open(filename_root+'_dna.txt')
#     read = open(read_filename)

#     orig_seq = orig.readline()
#     while orig_seq:
#         orig_seq = orig_seq.strip()
#         read_seq = read.readline().strip()
#         print(orig_seq)
#         print(read_seq)
#         ins, dels, subs, errs = get_errors(orig_seq, read_seq, True)
#         ins_rates.append(ins)
#         del_rates.append(dels)
#         sub_rates.append(subs)
#         err_rates.append(errs)
#         orig_seq = orig.readline()

#     orig.close()
#     read.close()

#     # Last seq is ignored as has larger number of errors if small sequence
#     print("AVGS: Ins: %s, dels: %s, subs: %s, err: %s" % (np.mean(ins_rates[:-1]), np.mean(del_rates[:-1]), np.mean(sub_rates[:-1]), np.mean(err_rates[:-1])))
