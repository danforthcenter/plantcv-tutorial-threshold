import os
import shutil
import pytest
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# inputs_list = [
#     ['modules', '01.thresholding.ipynb'],
#     ['modules', '02.edge-detection.ipynb'],
#     ['modules', '03.supervised-machine-learning.ipynb'],
#     ['modules', '04.unsupervised-machine-learning.ipynb'],
#     ['modules', '05.background-subtraction.ipynb'],
#     ['modules', '06.2d-thresholding.ipynb']
# ]
inputs_list = [
    os.path.join('modules', '01.thresholding.ipynb')
]


# ##########################
# Tests executing the notebook
# ##########################
@pytest.mark.parametrize('notebook', inputs_list)
def test_notebook(notebook, tmpdir):
    tmp = tmpdir.mkdir('sub')
    # Change working directory
    #os.chdir(dir)

    # Open the notebook
    with open(notebook, "r") as f:
        nb = nbformat.read(f, as_version=4)

    # Process the notebook
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": os.getwd()}})

    # Save the executed notebook
    out_nb = os.path.join(tmp, "executed_notebook.ipynb")
    with open(out_nb, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    assert os.path.exists(out_nb)

