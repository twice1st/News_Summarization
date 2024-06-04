from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, TrainingArguments, Seq2SeqTrainingArguments
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader
from datasets import load_metric



metric = load_metric("rouge")


model = AutoModelForSeq2SeqLM.from_pretrained(r"D:\summarization\checkpoint-7618")
model.to('cuda')


def inference(sentence):
  tokenizer = AutoTokenizer.from_pretrained("VietAI/vit5-base-vietnews-summarization")
  sentence = sentence + "</s>"
  encoding = tokenizer(sentence, return_tensors="pt")
  input_ids, attention_masks = encoding["input_ids"].to("cuda"), encoding["attention_mask"].to("cuda")
  outputs = model.generate(
      input_ids=input_ids, attention_mask=attention_masks,
      max_length=256,
      early_stopping=True
  )
  for output in outputs:
      line = tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
  return line

# sentence = "VietAI là tổ chức phi lợi nhuận với sứ mệnh ươm mầm tài năng về trí tuệ nhân tạo và xây dựng một cộng đồng các chuyên gia trong lĩnh vực trí tuệ nhân tạo đẳng cấp quốc tế tại Việt Nam."
# print(inference(sentence))