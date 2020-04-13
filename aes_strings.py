import argparse

# #####################################______KEY SCHEDULE_____###################################


def split_bytes(s):
    return [s[i:i + 8] for i in range(0, len(s), 8)]


def make_matrix(b):
    byts = split_bytes(b)
    # print(byts)

    matrix = [byts[i:i + 4] for i in range(0, len(byts), 4)]
    # print(matrix)
    matrix = zip(*matrix)
    # for each in matrix:
    #     print(each)

    # input -> 'abcdefghijklmnop'
    # output -> 'a, e, i, m'
    #           'b, f, j, n'
    #           'c, g, k, o'
    #           'd, h, l, p' for 128-bit key (k=4)
    return list(matrix)


def rotate_word(word, shift):
    # [a, b, c, d] -> [b, c, d, a]
    return word[shift:] + word[:shift]


def bin2int(b):
    return int(b, 2)


def hex2bin(h):
    return ''.join('{:04b}'.format(int(hex_digit, 16)) for hex_digit in h)


def S_box_encrypt(word):
    # input is in binary

    S = [['63', '7c', '77', '7b', 'f2', '6b', '6f', 'c5', '30', '01', '67', '2b', 'fe', 'd7', 'ab', '76'],
         ['ca', '82', 'c9', '7d', 'fa', '59', '47', 'f0', 'ad', 'd4', 'a2', 'af', '9c', 'a4', '72', 'c0'],
         ['b7', 'fd', '93', '26', '36', '3f', 'f7', 'cc', '34', 'a5', 'e5', 'f1', '71', 'd8', '31', '15'],
         ['04', 'c7', '23', 'c3', '18', '96', '05', '9a', '07', '12', '80', 'e2', 'eb', '27', 'b2', '75'],
         ['09', '83', '2c', '1a', '1b', '6e', '5a', 'a0', '52', '3b', 'd6', 'b3', '29', 'e3', '2f', '84'],
         ['53', 'd1', '00', 'ed', '20', 'fc', 'b1', '5b', '6a', 'cb', 'be', '39', '4a', '4c', '58', 'cf'],
         ['d0', 'ef', 'aa', 'fb', '43', '4d', '33', '85', '45', 'f9', '02', '7f', '50', '3c', '9f', 'a8'],
         ['51', 'a3', '40', '8f', '92', '9d', '38', 'f5', 'bc', 'b6', 'da', '21', '10', 'ff', 'f3', 'd2'],
         ['cd', '0c', '13', 'ec', '5f', '97', '44', '17', 'c4', 'a7', '7e', '3d', '64', '5d', '19', '73'],
         ['60', '81', '4f', 'dc', '22', '2a', '90', '88', '46', 'ee', 'b8', '14', 'de', '5e', '0b', 'db'],
         ['e0', '32', '3a', '0a', '49', '06', '24', '5c', 'c2', 'd3', 'ac', '62', '91', '95', 'e4', '79'],
         ['e7', 'c8', '37', '6d', '8d', 'd5', '4e', 'a9', '6c', '56', 'f4', 'ea', '65', '7a', 'ae', '08'],
         ['ba', '78', '25', '2e', '1c', 'a6', 'b4', 'c6', 'e8', 'dd', '74', '1f', '4b', 'bd', '8b', '8a'],
         ['70', '3e', 'b5', '66', '48', '03', 'f6', '0e', '61', '35', '57', 'b9', '86', 'c1', '1d', '9e'],
         ['e1', 'f8', '98', '11', '69', 'd9', '8e', '94', '9b', '1e', '87', 'e9', 'ce', '55', '28', 'df'],
         ['8c', 'a1', '89', '0d', 'bf', 'e6', '42', '68', '41', '99', '2d', '0f', 'b0', '54', 'bb', '16'], ]

    return [hex2bin(S[bin2int(w[:4])][bin2int(w[4:])]) for w in word]


def S_box_decrypt(word):
    # input is binary

    S = [['52', '09', '6a', 'd5', '30', '36', 'a5', '38', 'bf', '40', 'a3', '9e', '81', 'f3', 'd7', 'fb', ],
         ['7c', 'e3', '39', '82', '9b', '2f', 'ff', '87', '34', '8e', '43', '44', 'c4', 'de', 'e9', 'cb', ],
         ['54', '7b', '94', '32', 'a6', 'c2', '23', '3d', 'ee', '4c', '95', '0b', '42', 'fa', 'c3', '4e', ],
         ['08', '2e', 'a1', '66', '28', 'd9', '24', 'b2', '76', '5b', 'a2', '49', '6d', '8b', 'd1', '25', ],
         ['72', 'f8', 'f6', '64', '86', '68', '98', '16', 'd4', 'a4', '5c', 'cc', '5d', '65', 'b6', '92', ],
         ['6c', '70', '48', '50', 'fd', 'ed', 'b9', 'da', '5e', '15', '46', '57', 'a7', '8d', '9d', '84', ],
         ['90', 'd8', 'ab', '00', '8c', 'bc', 'd3', '0a', 'f7', 'e4', '58', '05', 'b8', 'b3', '45', '06', ],
         ['d0', '2c', '1e', '8f', 'ca', '3f', '0f', '02', 'c1', 'af', 'bd', '03', '01', '13', '8a', '6b', ],
         ['3a', '91', '11', '41', '4f', '67', 'dc', 'ea', '97', 'f2', 'cf', 'ce', 'f0', 'b4', 'e6', '73', ],
         ['96', 'ac', '74', '22', 'e7', 'ad', '35', '85', 'e2', 'f9', '37', 'e8', '1c', '75', 'df', '6e', ],
         ['47', 'f1', '1a', '71', '1d', '29', 'c5', '89', '6f', 'b7', '62', '0e', 'aa', '18', 'be', '1b', ],
         ['fc', '56', '3e', '4b', 'c6', 'd2', '79', '20', '9a', 'db', 'c0', 'fe', '78', 'cd', '5a', 'f4', ],
         ['1f', 'dd', 'a8', '33', '88', '07', 'c7', '31', 'b1', '12', '10', '59', '27', '80', 'ec', '5f', ],
         ['60', '51', '7f', 'a9', '19', 'b5', '4a', '0d', '2d', 'e5', '7a', '9f', '93', 'c9', '9c', 'ef', ],
         ['a0', 'e0', '3b', '4d', 'ae', '2a', 'f5', 'b0', 'c8', 'eb', 'bb', '3c', '83', '53', '99', '61', ],
         ['17', '2b', '04', '7e', 'ba', '77', 'd6', '26', 'e1', '69', '14', '63', '55', '21', '0c', '7d', ], ]

    return [hex2bin(S[bin2int(w[:4])][bin2int(w[4:])]) for w in word]


def R_con():
    vals = ['01', '02', '04', '08', '10', '20', '40', '80', '1B', '36']
    table = [[each] + ['00'] + ['00'] + ['00'] for each in vals]

    bin_table = [list(map(hex2bin, word)) for word in table]
    # for each in bin_table:
    #     print(each)
    return bin_table


def _xor(word1, word2):
    # input 2 columns

    return ['{:08b}'.format(int(i, 2) ^ int(j, 2)) for i, j in zip(word1, word2)]


def convert_and_pad(key):
    binary_key = ''.join(['{:08b}'.format(ord(char)) for char in key])

    if len(binary_key) in [128, 192, 256]:
        return binary_key

    if len(binary_key) < 128:
        print('length of the key is less than 128, padding key with zeros')
        binary_key += '0' * (128 - len(binary_key))

    elif len(binary_key) < 192:
        print('length of the is key less than 192, padding key with zeros')
        binary_key += '0' * (192 - len(binary_key))

    elif len(binary_key) < 256:
        print('length of the key is less than 256, padding key with zeros')
        binary_key += '0' * (256 - len(binary_key))
    elif len(binary_key) > 256:
        print('length of key is greater than 256 bits, truncating it to 256 bits')
        binary_key = binary_key[:256]
    else:
        pass

    return binary_key


def key_schedule(key):
    binary_key = convert_and_pad(key)

    k = len(binary_key) // 32
    # print(len(binary_key) // 32)

    def transform(word, xor_word, rcon_word):
        # rotate
        rotated = rotate_word(word, 1)
        # substitute
        subd = S_box_encrypt(rotated)
        # XOR
        xored = _xor(_xor(xor_word, subd), rcon_word)
        return xored

    final_keys_cols_ungrouped = []
    R_const_array = R_con()

    initial_key_state = make_matrix(binary_key)
    # print('I', initial_key_state)
    transposed = zip(*initial_key_state)

    [final_keys_cols_ungrouped.append(i) for i in transposed]

    for i in range(14 - k):
        word = final_keys_cols_ungrouped[-1]
        xor_word = final_keys_cols_ungrouped[-k]
        transformed_word = transform(word, xor_word, R_const_array[i])
        final_keys_cols_ungrouped.append(transformed_word)

        for j in range(k - 1):
            word = final_keys_cols_ungrouped[-1]
            xor_word = final_keys_cols_ungrouped[-k]
            final_keys_cols_ungrouped.append(_xor(word, xor_word))

    rounds = {128: 10, 192: 12, 256: 14}

    # return final_keys_cols_ungrouped
    # print(len(final_keys_cols_ungrouped), len(final_keys_cols_ungrouped[0]))
    final_keys_grouped = [list(zip(*final_keys_cols_ungrouped[i:i + 4])) for i in range(0, len(final_keys_cols_ungrouped), 4)]

    rnds = rounds[len(binary_key)]
    consider = final_keys_grouped[:rnds + 1]
    return consider

# #################------------------KEY SCHEDULE FINISH------------########################


def bin2hex(b):
    return ''.join(['{:x}'.format(int(b[i:i + 4], 2)) for i in range(0, len(b), 4)])


def xor_on_GF(a, b):
    return '{:08b}'.format(int(a, 2) ^ int(b, 2))


def galois_multiplication(b1, b2, irr):
    # INPUT b1 -> 8-bits BINARY, b2 -> 8-bits BINARY, irr -> DECIMAL
    # OUTPUT binary_val -> 8-bits BINARY

    # b1, b2 are in binary (little endianess)
    # irr is primitive / irreducible polynimial

    vect_a = [7 - i for i in range(8) if b1[i] == '1']
    vect_b = [7 - i for i in range(8) if b2[i] == '1']

    comb = sum([[i + j for i in vect_a] for j in vect_b], [])   # (a**i * a**j = a**(i+j))
    product = set([i for i in comb if comb.count(i) % 2 != 0])  # odd values because even values would get cancelled.
    if product == set():
        return '00000000'

    # if count of an element is even, then it would cancel itself
    while max(product) > 7:
        # modulo reduction using primitive polynomial unitl the maximum exponent is less than 8

        diff = max(product) - 8
        irr_ = [i + diff for i in irr]  # multiplying irr with a suitable number to counter highest degree
        product = product.symmetric_difference(set(irr_))   # product (XOR) multiplied irr polynomial

    result = [0] * 8
    for each in product:
        result[7 - each] = 1  # turning back exponents to binary

    binary_val = ''.join(map(str, result))
    return binary_val


def matrix_multiplication_over_2_256(m1, m2, irr):
    # INPUT m1 -> HEX, m2 -> BINARY
    # OUTPUT result -> BINARY

    # m1 will be RS or MDS
    # m2 will be a column matrix with bytes as elements

    if not len(m1[0]) == len(m2):
        raise Exception('Matrix order does not match for multiplication!')

    result = [['0' * 8 for i in range(len(m2[0]))] for j in range(len(m1))]
    for i in range(len(m1)):
        for j in range(len(m2[0])):
            for k in range(len(m1)):
                result[i][j] = xor_on_GF(result[i][j], galois_multiplication(hex2bin(m1[i][k]), m2[k][j], irr))
    return result


def encrypt(message, keys):

    # message = 'thisIsConfident.'
    # message should be 128 bit block -> 128/8 = 16 characters
    # print(len(message))

    binary_msg = ''.join(['{:08b}'.format(ord(char)) for char in message])
    # print('bin', binary_msg)
    state = make_matrix(binary_msg)

    initial_key = keys[0]

    # MIX COLUMN MATRIX
    MIX = [['02', '03', '01', '01'],
           ['01', '02', '03', '01'],
           ['01', '01', '02', '03'],
           ['03', '01', '01', '02']]

    ip_white_state = [_xor(initial_key[i], state[i]) for i in range(len(state))]

    # WHITENED STATE READY FOR 9 ROUNDS
    block = ip_white_state
    for i in range(1, len(keys) - 1):

        # [A0, A4, A8,  A12]
        # [A1, A5, A9,  A13]
        # [A2, A6, A10, A14]
        # [A3, A7, A11, A15]

        # SUB-BYTES
        # IT HAS NOTHING TO DO WITH ROW X COL ORDER
        subd_block = [S_box_encrypt(word) for word in block]

        # [S0, S4, S8,  S12]
        # [S1, S5, S9,  S13]
        # [S2, S6, S10, S14]
        # [S3, S7, S11, S15]

        # SHIFT ROWS
        shifted_block = [rotate_word(word, i) for i, word in enumerate(subd_block)]

        # [S0, S4, S8,  S12]
        # [S5, S9, S13,  S1]
        # [S10, S14, S2, S6]
        # [S15, S3, S7, S11]

        # MIX COLS
        # FOR THIS WE NEED TO TRANSPOSE TO ACCESS COLS AS ROW VECTORS FOR EASE
        block = list(zip(*shifted_block))
        # make each byte as  a vector for accessing
        block = [[[j] for j in i] for i in block]

        # [[S0],  [S5],  [S10], [S15]]
        # [[S4],  [S9],  [S14], [S3]]
        # [[S8],  [S13], [S2],  [S7]]
        # [[S12], [S1],  [S6],  [S11]]

        irr = [8, 4, 3, 1, 0]
        multiplied_block = [matrix_multiplication_over_2_256(MIX, word, irr) for word in block]

        # removing surrounded brackets added for multiplication purpose
        multiplied_block = [sum(i, []) for i in multiplied_block]

        # returning to columns state
        col_mixed_block = list(zip(*multiplied_block))

        key_added_block = [_xor(col_mixed_block[k], keys[i][k]) for k in range(len(col_mixed_block))]
        # print(i, key_added_block)

        block = key_added_block

    subd_block = [S_box_encrypt(word) for word in block]
    shifted_block = [rotate_word(word, i) for i, word in enumerate(subd_block)]

    final_key = keys[-1]
    op_white_state = [_xor(final_key[i], shifted_block[i]) for i in range(len(shifted_block))]

    cipher_block = zip(*op_white_state)

    cipher_bin = ''.join(sum(cipher_block, ()))
    # print(cipher_bin)
    # print(bin2hex(cipher_bin))

    return bin2hex(cipher_bin)


def decrypt(cipher, keys):

    # message = 'thisIsConfident.'
    # message should be 128 bit block -> 128/8 = 16 characters
    # print(len(message))

    binary_cipher = ''.join(['{:04b}'.format(int(hx, 16)) for hx in cipher])
    # print(binary_cipher)
    state = make_matrix(binary_cipher)

    initial_key = keys[0]

    # INVERSE MIX COLUMN MATRIX
    MIX = [['0e', '0b', '0d', '09'],
           ['09', '0e', '0b', '0d'],
           ['0d', '09', '0e', '0b'],
           ['0b', '0d', '09', '0e']]

    ip_white_state = [_xor(initial_key[i], state[i]) for i in range(len(state))]

    # WHITENED STATE READY FOR 9 ROUNDS
    block = ip_white_state
    for i in range(1, len(keys) - 1):

        # SHIFT ROWS
        shifted_block = [rotate_word(word, -i) for i, word in enumerate(block)]

        # SUB-BYTES
        # IT HAS NOTHING TO DO WITH ROW X COL ORDER
        subd_block = [S_box_decrypt(word) for word in shifted_block]

        # ADD-ROUND KEY
        key_added_block = [_xor(subd_block[k], keys[i][k]) for k in range(len(subd_block))]
        # print(i, key_added_block)

        # MIX COLS
        # FOR THIS WE NEED TO TRANSPOSE TO ACCESS COLS AS ROW VECTORS FOR EASE
        block = list(zip(*key_added_block))
        # make each byte as  a vector for accessing
        block = [[[j] for j in i] for i in block]

        irr = [8, 4, 3, 1, 0]
        multiplied_block = [matrix_multiplication_over_2_256(MIX, word, irr) for word in block]

        # removing surrounded brackets added for multiplication purpose
        multiplied_block = [sum(i, []) for i in multiplied_block]

        # returning to columns state
        col_mixed_block = list(zip(*multiplied_block))

        block = col_mixed_block

    final_key = keys[-1]

    shifted_block = [rotate_word(word, -i) for i, word in enumerate(block)]
    subd_block = [S_box_decrypt(word) for word in shifted_block]

    op_white_state = [_xor(final_key[i], subd_block[i]) for i in range(len(subd_block))]

    plain_block = zip(*op_white_state)

    plain_bin = ''.join(sum(plain_block, ()))
    # print('bin', plain_bin)

    def bin2str(b):
        return ''.join([chr(int(b[i:i + 8], 2)) for i in range(0, len(b), 8)])

    return bin2str(plain_bin)


parser = argparse.ArgumentParser()

parser.add_argument('-m', '--mode', choices=['enc', 'dec'], help='Encryption(enc) / Decryption(dec)')
parser.add_argument('-k', '--key', help='key for encryption / decryption')
parser.add_argument('-s', '--string', help='Enter the message for encryption or hex digest for decryption')

args = parser.parse_args()


key = args.key

keys = key_schedule(key)

message = args.string

if args.mode == 'enc':
    message = [message[i:i + 16] for i in range(0, len(message), 16)]
    if len(message[-1]) < 16:
        message[-1] += ' ' * (16 - len(message[-1]))

    cipher = ''.join(encrypt(msg_block, keys) for msg_block in message)
    print('CIPHER HEX:', cipher)

elif args.mode == 'dec':
    cipher_text = [message[i:i + 32] for i in range(0, len(message), 32)]
    plaintext = ''.join(decrypt(cipher_block, keys[:: -1]) for cipher_block in cipher_text)
    print('DECRYPTED:', plaintext.strip())
else:
    print('Invalid choice!')
