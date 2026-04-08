import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass


from collections import defaultdict

# 初始化一个默认值为 0 的字典
byte_counts = defaultdict(int)

# 模拟读取 UTF-8 编码的字节流
text_bytes = "hello".encode('utf-8') # b'\xe4\xbd\xa0\xe5\xa5\xbd'

for b in text_bytes:
    print(b)
    byte_counts[b] += 1

print(text_bytes)
print(type(text_bytes))
print(dict(byte_counts)) 
cur_pair = text_bytes[0:2]
print(dict(byte_counts)) 
for i in range(len(text_bytes) - 1):
    pair = text_bytes[i:i+2]
    
# 输出每个字节及其出现的频率