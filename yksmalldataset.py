import pandas as pd

import pyarrow as pa

# 目前windows无法正常运行保存！！！！

# 读取 parquet 文件
# df = pd.read_parquet(r'C:\hf_model\rm-static\data\test-00000-of-00001-8c7c51afc6d45980.parquet')
df = pd.read_parquet(r'~/hf_model/rm-static/data/test-00000-of-00001-8c7c51afc6d45980.parquet')

# 显示前几行数据
# print(df.head())
'''
PS C:\yk_repo\DeepSpeedExamples> python .\yksmalldataset.py
                                              prompt  ...                                           rejected
0  \n\nHuman: I am trying to write a fairy tale. ...  ...   And the prince and the princess both decide t...
1  \n\nHuman: What flowers should I grow to attra...  ...   In particular, it’s important to have a wide ...
2  \n\nHuman: How do I care for curly hair?\n\nAs...  ...   It’s an ongoing problem!  First, it’s importa...
3  \n\nHuman: How do I put out a kitchen fire?\n\...  ...                                    You’re welcome!
4  \n\nHuman: Why is sitting too close to the TV ...  ...             Are you going to ask me anything else?
'''


out_file = r'~/hf_model/rm-static/data/test-small.parquet'

# Create a parquet table from your dataframe

table = pa.Table.from_pandas(df.head(100))


# Write direct to your parquet file
pa.parquet.write_table(table, out_file)

out_file = r'~/hf_model/rm-static/data/train-small.parquet'

df = pd.read_parquet(r'~/hf_model/rm-static/data/train-00000-of-00001-2a1df75c6bce91ab.parquet')
# Create a parquet table from your dataframe

table = pa.Table.from_pandas(df.head(200))

# Write direct to your parquet file
pa.parquet.write_table(table, out_file)


