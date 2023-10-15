# from transformers import BloomTokenizerFast

# tokenizer = BloomTokenizerFast.from_pretrained("bloom-560m")
# print(tokenizer("Hello world")["input_ids"])

# tokenizer(" Hello world")["input_ids"]

# byte_sequence = b'\xe4\xbd\xa0\xe5\xa5\xbd'  # UTF-8 字节序列
# unicode_string = byte_sequence.decode("utf-8")  # 解码为 Unicode 字符串
# print(unicode_string)  # 输出 Unicode 字符串



# list1 = [1, 2, 2, 4, 5]
# list2 = [4, 2, 2, 7, 8]

# for i in list1:
#     if i in list2:
#         list2.remove(i)

# pass




import jieba
messages = jieba.cut("万里长城是中国古代劳动人民血汗的结晶和中国古代文化的象征和中华民族的骄傲",cut_all=False)   #精确模式
print ( "/ ".join(messages)) 