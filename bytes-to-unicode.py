import json
import re
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

def _tokenize(self, text):
    """Tokenize a string."""
    bpe_tokens = []
    for token in re.findall(self.pat, text):
        token = "".join(
            self.byte_encoder[b] for b in token.encode("utf-8")
        )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
        bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
    return bpe_tokens

def _convert_token_to_id(self, token):
    """Converts a token (str) in an id using the vocab."""
    return self.encoder.get(token, self.encoder.get(self.unk_token))

def _convert_id_to_token(self, index):
    """Converts an index (integer) in a token (str) using the vocab."""
    return self.decoder.get(index)

def convert_tokens_to_string(byte_decoder, tokens):
    """Converts a sequence of tokens (string) in a single string."""
    text = "".join(tokens)
    print(text)
    text_utf = bytearray([byte_decoder[c] for c in text]).decode("utf-8", 'ignore')
    return text_utf

from transformers import BloomTokenizerFast

byte_encoder = bytes_to_unicode()
byte_decoder = {v: k for k, v in byte_encoder.items()}
tokenizer = BloomTokenizerFast.from_pretrained("bloom-560m")
seg_id =  tokenizer("海运业雄踞全球之首，按吨位计占世界总数的１７％。")["input_ids"]

utf_vocab = {}
for key,value in tokenizer.vocab.items():
    key_unicode = convert_tokens_to_string(byte_decoder, key)
    utf_vocab[str(value)] = key_unicode

seg_id_ = "  ".join([utf_vocab[str(j)] for j in seg_id])
with open("tokenizer-unicode.json","w") as f:
    json.dump(utf_vocab,f, ensure_ascii=False)

