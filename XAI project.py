import random
import torch
from transformers import DistilBertTokenizer,DistilBertForSequenceClassification,DistilBertConfig
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.special import softmax
from lime.lime_text import LimeTextExplainer
#Accessing the data
data = pd.read_csv("downloads/imdb.csv")


#Taking a subset of the dataset
ss=data.sample(n=100, random_state=42).reset_index(drop=True)


#Converting the data to be compatible with the DistilBert model
ss['label']= ss['sentiment'].map({'positive':1,'negative':0})
#Setting up the LLM
tokenizer= DistilBertTokenizer.from_pretrained('lvwerra/distilbert-imdb')
configuration=DistilBertConfig.from_pretrained('lvwerra/distilbert-imdb',output_attentions=True)
model=DistilBertForSequenceClassification.from_pretrained('lvwerra/distilbert-imdb',config=configuration)


#Identifying the index of the future samples
samples=[]
for i in range(3):
    num=random.randint(0,len(ss['review'])-1)
    samples.append(num)
iid_att_mask=[]
ids=[]
sent=[]


#Drawing samples
for i in samples:
    sample = ss['review'][i]
    first_st= sample.replace('?','.').replace('!','.').split('.')[0]
    sent.append(first_st)
    new=tokenizer(first_st,padding=True,truncation=True,add_special_tokens=False)
    input_ids=new['input_ids']
    ids.append(input_ids)
    attention_mask=new['attention_mask']
    iid_att_mask.append((input_ids,attention_mask))
atts=[]


#Extracting the attention values
for i in iid_att_mask:
    returns=model(input_ids=torch.tensor(i[0]).reshape(1,-1).long(),attention_mask=torch.tensor(i[1]).reshape(1,-1).long(),output_attentions=True)
    atts.append(returns.attentions)


#Performing alpha-average
norm_atts=[]
for x in atts:
    lays_res=[]
    for k in x:
        squeezed=k.squeeze(0)
        heads_avg=squeezed.mean(dim=1)
        tokens_avg=heads_avg.mean(dim=0)
        lays_res.append(tokens_avg.detach().numpy())
    avg_layers=np.array(lays_res).mean(axis=0)
    norm=avg_layers/avg_layers.max()
    norm_atts.append(norm)


#Pairing the tokens with their repective attention weights
pairs=[]
unwanted=['br','/',',','-',"'",'"','(',')','#','>','<','*','&']
for i in zip(ids,norm_atts):
    tokens_decyphered=tokenizer.convert_ids_to_tokens(i[0])
    #Removing stopwords, symbols from the sentences
    pair=[(i,j) for i,j in zip(tokens_decyphered,i[1]) if i not in unwanted]
    pairs.append(pair)


#Visualization of the alpha-average method
fig,axs = plt.subplots(len(pairs),1,figsize=(12,6))
sns.heatmap(np.array([i[1] for i in pairs[0]]).reshape(1,-1),ax=axs[0])
tokens=[t for t,a in pairs[0]]
axs[0].set_xticks(range(len(tokens)))
axs[0].set_xticklabels(tokens,rotation=90)
sns.heatmap(np.array([i[1] for i in pairs[1]]).reshape(1,-1),ax=axs[1])
tokens=[t for t,a in pairs[1]]
axs[1].set_xticks(range(len(tokens)))
axs[1].set_xticklabels(tokens,rotation=90)
sns.heatmap(np.array([i[1] for i in pairs[2]]).reshape(1,-1),ax=axs[2])
tokens=[t for t,a in pairs[2]]
axs[2].set_xticks(range(len(tokens)))
axs[2].set_xticklabels(tokens,rotation=90)
plt.tight_layout()


#Defining the prediction function for LIME
def pred(l_s):
    new=tokenizer(l_s,padding=True,truncation=True,return_tensors='pt')
    model.eval()
    op=model(input_ids=new['input_ids'].long(),attention_mask=new['attention_mask'].long())
    probs=softmax(op.logits.detach().numpy(),axis=1)
    return probs


#Generating the LIME explanation for each sample sentence
explainer = LimeTextExplainer(class_names=['negative', 'positive'])
explanation1 = explainer.explain_instance(sent[0], pred, num_samples=2000, labels=[int(np.argmax(pred([sent[0]])))])
explanation2 = explainer.explain_instance(sent[1], pred, num_samples=2000, labels=[int(np.argmax(pred([sent[1]])))])
explanation3 = explainer.explain_instance(sent[2], pred, num_samples=2000, labels=[int(np.argmax(pred([sent[2]])))])


# Visualization
explanation1.as_pyplot_figure(label=int(np.argmax(pred([sent[0]]))))
explanation2.as_pyplot_figure(label=int(np.argmax(pred([sent[1]]))))
explanation3.as_pyplot_figure(label=int(np.argmax(pred([sent[2]]))))
plt.show()