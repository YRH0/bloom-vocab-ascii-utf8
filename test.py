# from transformers import BloomTokenizerFast

# tokenizer = BloomTokenizerFast.from_pretrained("bloom-560m")
# print(tokenizer("Hello world")["input_ids"])

# tokenizer(" Hello world")["input_ids"]

byte_sequence = b'\xe4\xbd\xa0\xe5\xa5\xbd'  # UTF-8 字节序列
unicode_string = byte_sequence.decode("utf-8")  # 解码为 Unicode 字符串
print(unicode_string)  # 输出 Unicode 字符串
