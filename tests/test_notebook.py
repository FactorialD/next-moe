"""Validate the deliverable and execute its core cells offline on the tiny profile."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import nbformat
import torch

ROOT = Path(__file__).parents[1]


def test_notebook_embedded_sources_and_syntax():
    notebook = nbformat.read(ROOT / 'notebooks/next_moe_0p2b_engram_0p1b.ipynb', as_version=4)
    nbformat.validate(notebook)
    sources = None
    for cell in notebook.cells:
        if cell.cell_type != 'code':
            continue
        text = '\n'.join(line for line in cell.source.splitlines() if not line.startswith('%'))
        tree = ast.parse(text)
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SOURCES' for t in node.targets):
                sources = ast.literal_eval(node.value)
    assert sources is not None
    for relative, content in sources.items():
        assert content == (ROOT / relative).read_text(), relative


def test_notebook_core_cells_offline(tmp_path):
    torch.set_num_threads(1)
    notebook = nbformat.read(ROOT / 'notebooks/next_moe_0p2b_engram_0p1b.ipynb', as_version=4)
    code_cells = [c.source for c in notebook.cells if c.cell_type == 'code']
    source = tmp_path / 'texts'
    for lang, text in [('uk', 'Україна має цікаву історію, її міста і люди. '),
                       ('en', 'History has many interesting stories and people. ')]:
        (source / lang).mkdir(parents=True)
        for i in range(6):
            (source / lang / f'{i}.txt').write_text((text + f'Article {i}. ') * 20)
    # Installation, Drive and plotting are Colab UI concerns. Core cells run unchanged
    # except explicit tiny/local/CPU test settings and writable path substitutions.
    import gc, shutil, time, math
    env = {'torch': torch, 'sys': sys, 'json': json, 'gc': gc, 'shutil': shutil, 'time': time, 'math': math, 'Path': Path}
    parameters = code_cells[1].replace("PROFILE = 'full'", "PROFILE = 'tiny'").replace('USE_DRIVE = True', 'USE_DRIVE = False')
    parameters = parameters.replace("'/content/next-moe-artifacts'", repr(str(tmp_path / 'artifacts')))
    parameters = parameters.replace("'/content/next-moe-code'", repr(str(tmp_path / 'code')))
    parameters = parameters.replace("'/content/next-moe-local'", repr(str(tmp_path / 'local')))
    exec(parameters, env)
    env.update(STEPS_OVERRIDE=2, SESSION_UPDATES=2, SEQ_LEN=8, GRAD_ACCUM=1, DOCUMENTS_PER_LANGUAGE=6,
               VALIDATION_DOCUMENTS=2, SOURCE='local', LOCAL_TEXT_DIR=str(source))
    # Embedded sources, smoke, data, model, training; index 7 plots; index 8 generates.
    for index in [2, 3, 4, 5, 6, 8, 9]:
        exec(code_cells[index], env)
    assert env['trainer'].step == 2
    assert (env['LOCAL_DIR'] / 'last.pt').exists()
    assert (env['RUN_DIR'] / 'weights.pt').exists()
    assert not (env['RUN_DIR'] / 'model.pt').exists()
    assert (env['DATA_DIR'] / 'documents.json').exists()
    saved_hashes = json.loads((env['RUN_DIR'] / 'source_hashes.json').read_text())
    assert all(saved_hashes[name] == hashlib.sha256(content.encode()).hexdigest()
               for name, content in env['SOURCES'].items())
