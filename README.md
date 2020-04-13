# aes_strings
Basic AES encryption &amp; decryption of strings

This code is about encrypting and decrypting strings of any length with 128 or 192 or 256 bit keys.
Any shorter length keys will be padded with zeros to next nearest availible key length and any key with length more than 256 bits will be truncated to 256 bits.

This code is written in pure python3 from scratch with using any modules but argparse (to execute this code from terminal)


# USAGE:

ENCRYPTION:
```
    $ python3 aes_strings.py -m enc -k your_secret_key -s 'encrypt this string'
                                      OR
    $ python3 aes_strings.py --mode enc --key your_secret_key --string 'encrypt this string'
```
DECRYPTION:
```
    $ python3 aes_strings.py -m dec -k your_secret_key -s 'decrypt this hex string'
                                      OR
    $ python3 aes_strings.py --mode dec --key your_secret_key --string 'decrypt this hex string'
```
