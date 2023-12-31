#!/bin/bash

# change this line if you are creating a new env
source /home/chetana/anaconda3/bin/activate
which python

# Use cat to create the requirements.txt file
cat <<EOL > requirements.txt
absl-py==0.9.0
adal==1.2.7
affine==2.3.0
alembic==1.7.7
anyio==3.6.2
appdirs==1.4.4
applicationinsights==0.11.10
argcomplete==2.1.2
argon2-cffi==21.3.0
argon2-cffi-bindings==21.2.0
asciitree==0.3.3
asn1crypto==0.24.0
astor==0.8.1
attrs==22.2.0
audioread==3.0.0
Automat==0.6.0
autoPyTorch==0.0.2
awscli==1.18.69
azure-common==1.1.28
azure-core==1.24.2
azure-graphrbac==0.61.1
azure-identity==1.7.0
azure-mgmt-authorization==2.0.0
azure-mgmt-containerregistry==10.0.0
azure-mgmt-core==1.3.2
azure-mgmt-keyvault==10.0.0
azure-mgmt-resource==21.1.0
azure-mgmt-storage==20.0.0
azureml==0.2.7
azureml-core==1.47.0
azureml-dataprep==4.5.7
azureml-dataprep-native==38.0.0
azureml-dataprep-rslex==2.11.4
azureml-dataset-runtime==1.47.0
azureml-opendatasets==1.47.0
azureml-telemetry==1.47.0
Babel==2.11.0
backcall==0.1.0
backports-datetime-fromisoformat==2.0.0
backports.tempfile==1.0
backports.weakref==1.0.post1
backports.zoneinfo==0.2.1
basemap==1.3.8
basemap-data==1.3.2
bcrypt==4.0.1
beautifulsoup4==4.6.0
bleach==3.1.0
blinker==1.4
blis==0.7.10
bokeh==2.3.3
Boruta==0.3
botocore==1.16.19
Bottleneck==1.3.7
bs4==0.0.1
cachetools==4.0.0
catalogue==1.0.2
cdo==1.3.5
certifi==2018.1.18
cffi==1.15.1
cftime==1.6.0
chardet==3.0.4
charset-normalizer==2.0.12
click==7.1.2
click-plugins==1.1.1
cligj==0.5.0
#cloud-init==23.1.2
cloudpickle==2.2.1
colorama==0.3.7
colorlover==0.3.0
#command-not-found==0.3
configobj==5.0.6
ConfigSpace==0.4.19
constantly==15.1.0
contextlib2==21.6.0
contextvars==2.4
cryptography==40.0.2
cufflinks==0.17.3
curlify==2.2.1
cycler==0.10.0
cymem==2.0.8
Cython==0.29.36
dask==2021.3.0
databricks-cli==0.17.7
#dataclasses==0.8
datefinder==0.7.0
dateparser==1.1.3
decorator==4.3.2
defusedxml==0.5.0
distributed==2021.3.0
distro==1.8.0
#distro-info===0.18ubuntu0.18.04.1
docker==5.0.3
docutils==0.14
dotnetcore2==3.1.23
#download-espa-order==2.2.5
earthengine-api==0.1.342
entrypoints==0.3
fasteners==0.18
filelock==3.4.1
Fiona==1.8.13.post1
Flask==2.0.3
fsspec==2022.1.0
funcy==2.0
fusepy==3.0.1
future==0.18.3
gast==0.2.2
gcloud==0.18.3
GDAL>2.2.3
gensim==3.8.3
geojson==2.5.0
geopandas==0.7.0
gitdb==4.0.9
GitPython==3.1.18
google-api-core==2.8.2
google-api-python-client==2.52.0
google-auth==1.35.0
google-auth-httplib2==0.1.0
google-auth-oauthlib==0.4.1
google-cloud-core==2.3.1
google-cloud-storage==2.0.0
google-crc32c==1.3.0
google-pasta==0.1.8
google-resumable-media==2.3.3
googleapis-common-protos==1.56.3
greenlet==2.0.2
grpcio==1.27.2
gunicorn==21.2.0
h5py==2.10.0
harmony-py==0.4.7
HeapDict==1.0.1
hpbandster==0.7.4
html5lib==0.999999999
htmldate==1.4.2
htmlmin==0.1.12
httplib2==0.21.0
huggingface-hub==0.4.0
humanfriendly==10.0
hyperlink==17.3.1
idna==2.10
ImageHash==4.3.1
imageio==2.5.0
imbalanced-learn==0.7.0
imblearn==0.0
immutables==0.19
importlib-metadata==4.8.3
importlib-resources==5.4.0
incremental==16.10.1
install==1.3.5
ipykernel==5.1.0
ipython==7.2.0
ipython-genutils==0.2.0
ipywidgets==7.4.2
isodate==0.6.1
itsdangerous==2.0.1
jedi==0.13.2
Jinja2==3.0.3
jmespath==0.9.3
joblib==1.0.1
json5==0.9.11
jsonpatch==1.16
jsonpickle==2.2.0
jsonpointer==1.10
jsonschema==3.2.0
jupyter==1.0.0
jupyter-client==7.1.2
jupyter-console==6.0.0
jupyter-core==4.9.2
jupyter-server==1.13.1
jupyterlab==3.2.9
jupyterlab-iframe==0.4.0
jupyterlab-server==2.10.3
Keras==2.3.1
Keras-Applications==1.0.8
Keras-Preprocessing==1.1.0
keyring==10.6.0
keyrings.alt==3.0
kiwisolver==1.1.0
kmodes==0.12.2
knack==0.10.1
#language-selector==0.1
liac-arff==2.5.0
librosa==0.9.2
lightgbm==3.3.5
llvmlite==0.36.0
loader==2017.9.11
locket==1.0.0
lxml==4.9.2
Mako==1.1.6
Markdown==3.2.1
MarkupSafe==2.0.1
matplotlib==3.3.4
minio==7.1.15
missingno==0.5.2
mistune==0.8.4
mlflow==1.23.1
mlxtend==0.19.0
msal==1.22.0
msal-extensions==0.3.1
msgpack==1.0.5
msrest==0.7.1
msrestazure==0.6.4
multimethod==1.5
munch==2.5.0
murmurhash==1.0.10
nbclassic==0.3.5
nbconvert==5.4.1
nbformat==4.4.0
ndg-httpsclient==0.5.1
nest-asyncio==1.5.6
netCDF4==1.6.2
netifaces==0.10.4
networkx==2.5.1
nltk==3.6.7
notebook==5.7.4
numba==0.53.0
numcodecs==0.9.1
numexpr==2.6.4
numpy==1.19.5
oauth2client==4.1.3
oauthlib==3.1.0
olefile==0.45.1
opencv-python==4.7.0.68
openml==0.14.0
opt-einsum==3.2.0
packaging==21.3
#PAM==0.4.2
pandas==1.1.5
pandas-profiling==3.1.0
pandocfilters==1.4.2
paramiko==2.12.0
parso==0.3.4
partd==1.2.0
pathlib==1.0.1
pathspec==0.9.0
patsy==0.5.3
pexpect==4.6.0
phik==0.12.0
pickleshare==0.7.5
Pillow==8.4.0
Pint==0.17
pkginfo==1.9.6
plac==1.1.3
plotly==5.17.0
pockets==0.9.1
Polygon3==3.0.9.1
pooch==1.6.0
portalocker==2.7.0
preshed==3.0.9
progressbar2==3.55.0
prometheus-client==0.5.0
prometheus-flask-exporter==0.22.4
prompt-toolkit==2.0.8
protobuf==3.19.6
psutil==5.9.4
ptyprocess==0.6.0
py4j==0.10.9.5
pyarrow==6.0.1
pyasn1==0.4.2
pyasn1-modules==0.2.1
pycaret==2.3.10
pycparser==2.21
pycrate==0.6.0
pycrypto==2.6.1
pydantic==1.9.2
pyDeprecate==0.3.2
pyEddyTracker==3.2.0
pyEddyTrackerSample==0.1.0
Pygments==2.3.1
#PyGObject==3.26.1
PyJWT==2.4.0
pyLDAvis==3.2.2
PyNaCl==1.5.0
pynisher==0.6.4
pynndescent==0.5.10
pyod==1.1.0
pyOpenSSL==17.5.0
pyparsing==2.4.2
pyproj==2.5.0
Pyro4==4.82
pyrsistent==0.18.0
pyserial==3.4
pyshp==2.1.0
PySocks==1.7.1
pyspark==3.2.4
pystac==0.5.6
pystac-client==0.1.1
python-apt==1.6.5+ubuntu0.5
python-dateutil==2.8.2
python-debian==0.1.32
python-dotenv==0.20.0
python-utils==3.5.2
pytz==2019.2
pytz-deprecation-shim==0.1.0.post0
PyWavelets==1.0.3
pyxdg==0.25
PyYAML==5.4.1
pyzmq==17.1.2
qtconsole==4.4.3
querystring-parser==1.2.4
rasterio==1.1.3
regex==2023.8.8
requests==2.27.1
requests-oauthlib==1.3.0
requests-unixsocket==0.1.5
resampy==0.4.2
roman==2.0.0
rsa==4.0
s3transfer==0.3.3
sacremoses==0.0.53
scikit-image==0.15.0
scikit-learn==0.23.2
scikit-plot==0.3.7
scipy==1.5.4
scour==0.36
screen-resolution-extra==0.0.0
seaborn==0.11.2
SecretStorage==2.3.1
Send2Trash==1.5.0
serpent==1.41
service-identity==16.0.0
shap
Shapely==1.7.0
six==1.14.0
sklearn==0.0
smart-open==6.4.0
smmap==5.0.0
sniffio==1.2.0
snuggs==1.4.7
sortedcontainers==2.4.0
sos==4.3
soundfile==0.12.1
spacy==2.3.9
sphinxcontrib-napoleon==0.7
SQLAlchemy==1.4.49
sqlparse==0.4.4
srsly==1.0.7
ssh-import-id==5.7
statsmodels==0.12.2
systemd-python==234
tables==3.4.2
tabulate==0.8.10
tangled-up-in-unicode==0.1.0
tbb==2021.10.0
tblib==1.7.0
tenacity==8.2.2
tensorboard==2.1.1
tensorboard-logger==0.1.0
tensorflow==2.1.0
tensorflow-estimator==2.1.0
tensorflow-gpu==2.1.0
termcolor==1.1.0
terminado==0.12.1
testpath==0.4.2
textblob==0.17.1
thinc==7.4.6
threadpoolctl==3.1.0
tifftools==1.3.9
tokenizers==0.12.1
toolz==0.8.2
torch==1.10.1
torchmetrics==0.8.2
torchvision==0.11.2
tornado==6.1
tornado-proxy-handlers==0.0.5
tqdm==4.64.1
traitlets==4.3.2
transformers==4.18.0
Twisted==17.9.0
typing_extensions==4.1.1
tzdata==2022.7
tzlocal==4.2
ubuntu-drivers-common==0.0.0
ufw==0.35
umap-learn==0.5.4
unattended-upgrades==0.1
uritemplate==4.1.1
urllib3==1.26.15
virtualenv==16.6.1
visions==0.7.4
wasabi==0.10.1
wcwidth==0.1.7
webencodings==0.5.1
websocket-client==1.3.1
Werkzeug==2.0.3
wget==3.2
widgetsnbextension==3.4.2
wordcloud==1.9.2
wrapt==1.12.1
xarray==0.10.2
xkit==0.0.0
xmltodict==0.13.0
yellowbrick==1.3.post1
zarr==2.8.3
zict==2.1.0
zipp==3.6.0
zope.interface==4.3.2
EOL

python -m pip install -r requirements.txt
# clean up
rm requirements.txt

