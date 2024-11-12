import os
import shutil
import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

project_root = os.getcwd()
inputs_list = [
    [os.path.join(project_root, 'modules'), '01.thresholding.ipynb'],
    [os.path.join(project_root, 'modules'), '02.edge-detection.ipynb'],
    [os.path.join(project_root, 'modules'), '03.supervised-machine-learning.ipynb'],
    [os.path.join(project_root, 'modules'), '04.unsupervised-machine-learning.ipynb'],
    [os.path.join(project_root, 'modules'), '05.background-subtraction.ipynb'],
    [os.path.join(project_root, 'modules'), '06.2d-thresholding.ipynb']
]


# ##########################
# Tests executing the notebook
# ##########################
@pytest.mark.parametrize('dir,notebook', inputs_list)
def test_notebook(dir, notebook, tmpdir):
    tmp = tmpdir.mkdir('sub')
    # Change working directory
    #os.chdir(dir)

    # Open the notebook
    with open(os.path.join(dir, notebook), "r") as f:
        nb = nbformat.read(f, as_version=4)

    # Process the notebook
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": dir}})

    # Save the executed notebook
    out_nb = os.path.join(tmp, "executed_notebook.ipynb")
    with open(out_nb, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    assert os.path.exists(out_nb)

