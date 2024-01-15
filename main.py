from fastapi import FastAPI


app = FastAPI()


@app.get("")
def pong():
    return {"status": "working"}
