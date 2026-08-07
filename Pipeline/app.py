from detection import app as detection_app
from generation import app as generation_app
from staging import app as staging_app

app = staging_app
app.include(detection_app)
app.include(generation_app)
