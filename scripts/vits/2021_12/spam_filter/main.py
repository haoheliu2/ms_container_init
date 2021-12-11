#!/usr/bin/env python
# coding: utf-8

# In[102]:


import pandas as pd
import os

FILTER_OUT_SPAM_3 = True
FILTER_OUT_SPAM_2_THREE_TIMES = True
FILTER_ZERO_RATIO_THRESHOLD=0.01
DATA_PATH="./data"

for fpath in os.listdir(DATA_PATH):
       if(".xls" not in fpath or fpath[0] == "."): continue
       print("\n-----------------------------------")
       fpath = os.path.join(DATA_PATH, fpath)
       print("Start processing: ", fpath)
       sh_name = pd.read_excel(fpath, sheet_name = "Analysis Summary-GeneralSe")
       names = [name.strip() for name in sh_name.at[4, "Unnamed: 3"].split("vs.")]
       baseline_name = names[0]
       another_name = names[1]

       print("Process", fpath)
       sh_spam = pd.read_excel(fpath, sheet_name = "SpamCases")
       sh_details = pd.read_excel(fpath, sheet_name = "Details")
       origin_length = len(sh_details)
       sh_details.drop_duplicates(["LE name","Domain","Script","Voice name","Wave order"],"first",inplace=True)
       print("Spot %d duplicate line in sheet: Details" % (origin_length - len(sh_details)))
       score_no_filter_no_spam = sh_details.groupby("Domain").get_group("GeneralSentence").groupby("Voice name").get_group(another_name)["Score"].mean()
       score_no_filter_with_spam = sh_details.groupby("Voice name").get_group(another_name)["Score"].mean()
       print("Mean CMOS score without filter is %f. Calculated with spam: " % score_no_filter_with_spam)
       print("Mean CMOS score without filter is %f. Calculated without spam: " % score_no_filter_no_spam)


       # In[103]:


       all_judgers = sh_spam["JudgeId"].tolist()
       filted_type_1 = sh_spam[sh_spam["Score(+-3)"]>0]["JudgeId"].tolist()
       print("Judgers: ",filted_type_1," scoring on spam case is +/-3")
       filted_type_2 = sh_spam[sh_spam["Score(+-2)"]>=3]["JudgeId"].tolist()
       print("Judgers: ",filted_type_2," scoring on spam case is +/-2 for 3 times or more")
       filted_type_3 = sh_spam[sh_spam["Percent(Score:0)"].str.strip("%").astype(float)/100 < FILTER_ZERO_RATIO_THRESHOLD]["JudgeId"].tolist()
       print("Judgers: ", filted_type_3, "zero score ratio on spam case is < %.1f percent" % (FILTER_ZERO_RATIO_THRESHOLD * 100))
       excluded_judger = []
       excluded_judger.extend(filted_type_1+filted_type_2+filted_type_3)
       excluded_judger = list(set(excluded_judger))
       remain_judgers = [x for x in all_judgers if x not in excluded_judger]
       print("--> Finally we only consider these judgers:", remain_judgers)


       # In[104]:


       sh_filtered_details = sh_details[sh_details["LE name"].isin(remain_judgers)]
       score_filtered_no_spam = sh_filtered_details.groupby("Domain").get_group("GeneralSentence").groupby("Voice name").get_group(another_name)["Score"].mean()
       score_filtered_with_spam = sh_filtered_details.groupby("Voice name").get_group(another_name)["Score"].mean()
       print("Mean CMOS score with filter is %f. Calculated with spam." % score_filtered_with_spam)
       print("Mean CMOS score with filter is %f. Calculated without spam." % score_filtered_no_spam)


       # In[105]:
       data = pd.DataFrame(
              {"Mean CMOS with spam no filter":[score_no_filter_with_spam], 
              "Mean CMOS with spam with filter":[score_filtered_with_spam], 
              "Mean CMOS no spam no filter":[score_no_filter_no_spam], 
              "Mean CMOS no spam with filter":[score_filtered_no_spam], 
              }
              )
       with pd.ExcelWriter(fpath, mode='a', engine='openpyxl', if_sheet_exists="replace") as writer:
              data.to_excel(writer, sheet_name="filted_result")

print("Done")

