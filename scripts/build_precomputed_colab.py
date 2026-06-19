"""Build the self-contained Colab notebook for paper tile statistics."""

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'notebooks' / 'SmokeyNet_PaperDataset_precomputed.ipynb'


def source(text):
    return text.strip('\n').splitlines(keepends=True)


def markdown(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': source(text)}


def code(text):
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': source(text),
    }


repo_patch = subprocess.run(
    ['git', 'diff', 'master', '--', 'src/dynamic_dataloader.py', 'src/main.py'],
    cwd=ROOT,
    check=True,
    capture_output=True,
    text=True,
).stdout
if not repo_patch:
    raise SystemExit('No source difference from master found; update the notebook setup cell.')


cells = [
    markdown(r'''
# SmokeyNet con el dataset original y etiquetas oficiales por tile

Este notebook entrena la arquitectura `ResNet34 + LSTM + SpatialViT` con los splits del articulo y `labels_stats_90overlap.pkl`. Las positivas sin estadisticas oficiales se excluyen del entrenamiento; las negativas conservan sus 45 tiles a cero.

Ejecuta primero `diagnostic`. Cambia a `full` solo cuando el smoke test y una epoca completa terminen correctamente.
'''),
    code(r'''
# 1. Configuracion y persistencia
from google.colab import drive
from pathlib import Path
import os

drive.mount('/content/drive')

RUN_MODE = 'diagnostic'  # 'diagnostic' o 'full'
REPO_URL = 'https://github.com/PedroHT8/SmokeyNet.git'
REPO_DIR = Path('/content/SmokeyNet')
RAW_ROOT = Path('/content/data/raw_images')
DRIVE_ROOT = Path('/content/drive/MyDrive/TFM_SmokeyNet_PaperDataset')
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)

print('Modo:', RUN_MODE)
print('Checkpoints:', DRIVE_ROOT)
'''),
    code(f'''
# 2. Clonar el fork e instalar temporalmente el soporte de etiquetas precomputadas
import subprocess
import shutil

if not (REPO_DIR / '.git').exists():
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    subprocess.run(['git', 'clone', REPO_URL, str(REPO_DIR)], check=True)

main_text = (REPO_DIR / 'src/main.py').read_text(encoding='utf-8')
if '--tile-label-stats-path' not in main_text:
    patch_text = {repo_patch!r}
    patch_path = Path('/tmp/smokeynet_precomputed_labels.patch')
    patch_path.write_text(patch_text, encoding='utf-8')
    subprocess.run(
        ['git', 'apply', '--ignore-space-change', '--ignore-whitespace', str(patch_path)],
        cwd=REPO_DIR,
        check=True,
    )
    print('Parche de etiquetas precomputadas aplicado.')
else:
    print('El fork ya incluye etiquetas precomputadas; no se aplica parche.')

subprocess.run(['git', 'diff', '--check'], cwd=REPO_DIR, check=True)
'''),
    code(r'''
# 3. Dependencias compatibles con el fork adaptado
import sys
import subprocess

packages = [
    'pytorch-lightning>=2.2,<2.7',
    'torchmetrics>=1.3,<1.10',
    'transformers',
    'efficientnet-pytorch',
    'gtrxl-torch',
    'opencv-python-headless',
    'scikit-learn',
    'tensorboard',
]
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *packages], check=True)

import torch
import pytorch_lightning as pl
print('Python:', sys.version.split()[0])
print('Torch:', torch.__version__)
print('Lightning:', pl.__version__)
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO DISPONIBLE')
if not torch.cuda.is_available():
    raise RuntimeError('Activa un runtime con GPU antes de continuar.')
'''),
    code(r'''
# 4. Descargar las 270 secuencias de los splits oficiales
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

SPLIT_DIR = REPO_DIR / 'data/final_split'
split_paths = {
    'train': SPLIT_DIR / 'train_images_final.txt',
    'val': SPLIT_DIR / 'val_images_final.txt',
    'test': SPLIT_DIR / 'test_images_final.txt',
}

def normalized_name(line):
    parts = line.strip().replace('\\', '/').split('/')
    return f'{parts[-2]}/{Path(parts[-1]).stem}'

required_images = {
    normalized_name(line)
    for split_path in split_paths.values()
    for line in split_path.read_text().splitlines()
    if line.strip()
}
fires = sorted({name.split('/')[0] for name in required_images})
RAW_ROOT.mkdir(parents=True, exist_ok=True)

def download_fire(fire):
    fire_dir = RAW_ROOT / fire
    if fire_dir.exists() and any(fire_dir.glob('*.jpg')):
        return fire, 'presente'
    archive = Path('/content') / f'{fire}.tgz'
    url = f'https://cdn.hpwren.ucsd.edu/HPWREN-FIgLib-Data/Tar/{fire}.tgz'
    subprocess.run(['wget', '-q', '--show-progress', '-O', str(archive), url], check=True)
    subprocess.run(['tar', '-xzf', str(archive), '-C', str(RAW_ROOT)], check=True)
    archive.unlink(missing_ok=True)
    return fire, 'descargado'

print(f'Descargando/verificando {len(fires)} incendios y {len(required_images)} imagenes...')
with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(download_fire, fire) for fire in fires]
    for index, future in enumerate(as_completed(futures), 1):
        fire, status = future.result()
        if index % 10 == 0 or index == len(futures):
            print(f'{index}/{len(futures)}: {fire} ({status})')

missing_images = sorted(
    name for name in required_images if not (RAW_ROOT / f'{name}.jpg').exists()
)
print('Imagenes requeridas:', len(required_images))
print('Imagenes ausentes:', len(missing_images))
if missing_images:
    raise FileNotFoundError(f'Faltan imagenes del split. Primeras: {missing_images[:10]}')
'''),
    code(r'''
# 5. Preflight de etiquetas y balance
import pickle

STATS_PATH = REPO_DIR / 'data/label_stats/labels_stats_90overlap.pkl'
METADATA_PATH = REPO_DIR / 'data/metadata.pkl'
with STATS_PATH.open('rb') as handle:
    tile_stats = pickle.load(handle)

train_images = [
    normalized_name(line)
    for line in split_paths['train'].read_text().splitlines()
    if line.strip()
]
invalid = [name for name, counts in tile_stats.items() if len(counts) != 45]
positive = [name for name in train_images if '+' in name]
negative = [name for name in train_images if '+' not in name]
covered = [name for name in positive if name in tile_stats]
missing_positive = [name for name in positive if name not in tile_stats]
positive_tiles = sum(sum(value > 250 for value in tile_stats[name]) for name in covered)
negative_tiles = (len(covered) + len(negative)) * 45 - positive_tiles

summary = {
    'train_original': len(train_images),
    'positive_covered': len(covered),
    'negative_retained': len(negative),
    'positive_omitted': len(missing_positive),
    'train_effective': len(covered) + len(negative),
    'positive_tiles': positive_tiles,
    'negative_tiles': negative_tiles,
    'negative_positive_ratio': negative_tiles / positive_tiles,
}
print(summary)
assert not invalid
assert summary['train_effective'] == 10639
assert summary['positive_covered'] == 5013
assert 35 < summary['negative_positive_ratio'] < 38
print('Preflight correcto.')
'''),
    code(r'''
# 6. Smoke test del DataLoader antes de usar horas de GPU
import sys
sys.path.insert(0, str(REPO_DIR / 'src'))
from dynamic_dataloader import DynamicDataModule

dm = DynamicDataModule(
    omit_list=['omit_no_xml'],
    raw_data_path=str(RAW_ROOT),
    labels_path=None,
    tile_label_stats_path=str(STATS_PATH),
    metadata_path=str(METADATA_PATH),
    train_split_path=str(split_paths['train']),
    val_split_path=str(split_paths['val']),
    test_split_path=str(split_paths['test']),
    load_images_from_split=True,
    batch_size=1,
    num_workers=0,
    series_length=2,
    resize_dimensions=(1392, 1856),
    crop_height=1040,
    tile_dimensions=(224, 224),
    tile_overlap=20,
    smoke_threshold=250,
    resize_crop_augment=False,
    blur_augment=False,
    color_augment=False,
    brightness_contrast_augment=False,
)
dm.setup()
batch = next(iter(dm.train_dataloader()))
_, images, labels, _, image_gt, omit_mask = batch
print('images:', tuple(images.shape))
print('tile labels:', tuple(labels.shape), 'positive tiles:', int(labels.sum()))
print('image gt:', image_gt.tolist(), 'omit mask:', omit_mask.tolist())
assert images.shape[1:] == (45, 2, 3, 224, 224)
assert labels.shape[1] == 45
print('Smoke test correcto.')
'''),
    code(r'''
# 7. Argumentos de entrenamiento
import shlex

EXPERIMENT_NAME = f'smokeynet_paper_precomputed_{RUN_MODE}'
MAX_EPOCHS = 2 if RUN_MODE == 'diagnostic' else 25

base_args = [
    sys.executable, '-u', 'src/main.py',
    '--experiment-name', EXPERIMENT_NAME,
    '--experiment-description', 'SmokeyNet paper dataset with official precomputed tile statistics.',
    '--raw-data-path', str(RAW_ROOT),
    '--labels-path', '/content/unused_labels',
    '--tile-label-stats-path', str(STATS_PATH),
    '--metadata-path', str(METADATA_PATH),
    '--train-split-path', str(split_paths['train']),
    '--val-split-path', str(split_paths['val']),
    '--test-split-path', str(split_paths['test']),
    '--load-images-from-split',
    '--omit-list', 'omit_no_xml',
    '--error-as-eval-loss',
    '--model-type-list', 'RawToTile_ResNet', 'TileToTile_LSTM', 'TileToTileImage_SpatialViT',
    '--use-image-preds',
    '--series-length', '2',
    '--resize-height', '1392', '--resize-width', '1856', '--crop-height', '1040',
    '--tile-size', '224', '--tile-overlap', '20', '--smoke-threshold', '250',
    '--no-resize-crop-augment',
    '--batch-size', '1', '--accumulate-grad-batches', '32', '--num-workers', '0',
    '--tile-loss-type', 'bce', '--bce-pos-weight', '40', '--image-pos-weight', '5',
    '--optimizer-type', 'SGD', '--learning-rate', '0.001', '--optimizer-weight-decay', '0.001',
    '--min-epochs', '1', '--max-epochs', str(MAX_EPOCHS),
    '--no-early-stopping', '--gradient-clip-val', '1.0',
    '--tile-embedding-size', '1000', '--backbone-size', 'small',
]
if RUN_MODE == 'diagnostic':
    base_args.append('--no-stochastic-weight-avg')

print(' '.join(shlex.quote(str(arg)) for arg in base_args))
'''),
    code(r'''
# 8. Entrenar. La salida se muestra en directo y no queda oculta en capture_output.
env = os.environ.copy()
env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
env['PYTHONPATH'] = str(REPO_DIR / 'src')

subprocess.run(base_args, cwd=REPO_DIR, env=env, check=True)
print('Entrenamiento terminado correctamente.')
'''),
    code(r'''
# 9. Localizar y evaluar el mejor checkpoint explicitamente
checkpoint_dir = REPO_DIR / 'lightning_logs' / EXPERIMENT_NAME
best_candidates = sorted(
    [p for p in checkpoint_dir.glob('version_*/checkpoints/*.ckpt') if p.name != 'last.ckpt'],
    key=lambda p: p.stat().st_mtime,
)
last_candidates = sorted(
    checkpoint_dir.glob('version_*/checkpoints/last.ckpt'),
    key=lambda p: p.stat().st_mtime,
)
if not best_candidates:
    raise FileNotFoundError('No se encontro el best checkpoint.')

BEST_CKPT = best_candidates[-1]
LAST_CKPT = last_candidates[-1] if last_candidates else None
print('BEST_CKPT:', BEST_CKPT)
print('LAST_CKPT:', LAST_CKPT)

eval_args = base_args.copy()
EVAL_EXPERIMENT_NAME = EXPERIMENT_NAME + '_best_test'
eval_args[eval_args.index('--experiment-name') + 1] = EVAL_EXPERIMENT_NAME
eval_args += ['--checkpoint-path', str(BEST_CKPT), '--is-test-only']
eval_proc = subprocess.run(
    eval_args,
    cwd=REPO_DIR,
    env=env,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
)
print(eval_proc.stdout[-30000:])
if eval_proc.returncode != 0:
    raise RuntimeError(f'La evaluacion fallo con returncode={eval_proc.returncode}.')
print('Evaluacion del best checkpoint terminada.')

# TensorBoard provides a second, machine-readable route to the test metrics.
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

eval_event_files = sorted(
    (REPO_DIR / 'lightning_logs' / EVAL_EXPERIMENT_NAME).glob('version_*/events.out.tfevents.*'),
    key=lambda path: path.stat().st_mtime,
)
TEST_METRICS = {}
if eval_event_files:
    accumulator = EventAccumulator(str(eval_event_files[-1]))
    accumulator.Reload()
    for tag in accumulator.Tags().get('scalars', []):
        if tag.startswith('test/') and accumulator.Scalars(tag):
            TEST_METRICS[tag] = accumulator.Scalars(tag)[-1].value

print('Metricas test recuperadas de TensorBoard:')
for name, value in sorted(TEST_METRICS.items()):
    print(f'{name}: {value:.6f}')
if not TEST_METRICS:
    print('AVISO: TensorBoard no contiene escalares test; revisa la salida anterior.')
'''),
    code(r'''
# 10. Guardar checkpoints y logs en Drive
import shutil
import json
from datetime import datetime

destination = DRIVE_ROOT / f'{EXPERIMENT_NAME}_{datetime.now():%Y%m%d_%H%M%S}'
destination.mkdir(parents=True, exist_ok=True)
shutil.copy2(BEST_CKPT, destination / 'best.ckpt')
if LAST_CKPT is not None:
    shutil.copy2(LAST_CKPT, destination / 'last.ckpt')

log_source = BEST_CKPT.parents[1]
for event_file in log_source.glob('events.out.tfevents.*'):
    shutil.copy2(event_file, destination / event_file.name)
if eval_event_files:
    shutil.copy2(eval_event_files[-1], destination / 'best_test_events.tfevents')
(destination / 'best_test_metrics.json').write_text(
    json.dumps(TEST_METRICS, indent=2, sort_keys=True),
    encoding='utf-8',
)
print('Resultados guardados en:', destination)
'''),
]


notebook = {
    'cells': cells,
    'metadata': {
        'accelerator': 'GPU',
        'colab': {'name': OUTPUT.name, 'provenance': []},
        'kernelspec': {'display_name': 'Python 3', 'name': 'python3'},
        'language_info': {'name': 'python'},
    },
    'nbformat': 4,
    'nbformat_minor': 5,
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print(OUTPUT)
