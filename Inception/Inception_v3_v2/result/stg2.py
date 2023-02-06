import pandas as pd

df_stage1 = pd.read_csv('stage1_submission.csv')
df_stage2 = pd.read_csv('stage2_submission.csv')
df_stage2.image =df_stage2.image.map(lambda x: 'test_stg2/'+x)
df_final = pd.concat((df_stage1, df_stage2), axis=0, ignore_index=True)
df_final.to_csv('final_submission.csv', index=False)