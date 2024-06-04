from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
import summarization
from fastapi import FastAPI, Form
from typing_extensions import Annotated

app = FastAPI()
@app.get("/")
async def read_root():
    return {'helloworld'}


@app.post('/summarization')
async def inference(txt: Annotated[str, Form()]):
    print("Enter news to summary !!!")
    respone = summarization.inference(txt)
    return {"result of summarization: " : respone}

 