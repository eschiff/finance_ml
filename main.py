from fastapi import FastAPI

from routes.monitor import monitor_router
from routes.exceptions import ServiceHTTPException, handle_service_http_exception

app = FastAPI()

app.add_exception_handler(ServiceHTTPException, handle_service_http_exception)
app.include_router(monitor_router)


@app.on_event("startup")
async def startup():
    # perform startup tasks
    pass


@app.on_event("shutdown")
async def shutdown():
    # perform shutdown tasks
    pass
