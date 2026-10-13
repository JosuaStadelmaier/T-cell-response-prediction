"""
Tests that the model of this package is the model of the repository.

finet/models.py is a copy of the model classes of code/models.py, so that the saved models
load into the same module structure. This test fails if one of the two copies changes,
which would silently change the predictions or break the loading of the saved models.
"""
import ast
import hashlib
import os

import pytest

PACKAGE_MODELS = os.path.join(os.path.dirname(__file__), os.pardir, 'finet', 'models.py')
REPOSITORY_MODELS = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'code', 'models.py')

SHARED_CLASSES = ['PositionalEncoding', 'TransformerEmbedding', 'LearnedPosEmbedding',
                  'MHC_presentation', 'PeptideTransformer']

SAVED_MODELS = {
    'FINET_I_II_pre_I_0_1.tar': 'b5c05b01dde8d15f',
    'FINET_I_II_pre_II_0_1.tar': '992e91fc4408d56d',
}


def class_definitions(file_path):
    source = open(file_path).read()
    lines = source.split('\n')
    return {node.name: '\n'.join(lines[node.lineno - 1:node.end_lineno])
            for node in ast.parse(source).body if isinstance(node, ast.ClassDef)}


@pytest.mark.skipif(not os.path.exists(REPOSITORY_MODELS),
                    reason='the repository implementation is only available in the repository')
@pytest.mark.parametrize('class_name', SHARED_CLASSES)
def test_the_model_classes_are_the_ones_of_the_repository(class_name):
    package_classes = class_definitions(PACKAGE_MODELS)
    repository_classes = class_definitions(REPOSITORY_MODELS)

    assert class_name in package_classes
    assert class_name in repository_classes
    assert package_classes[class_name] == repository_classes[class_name]


def test_the_package_defines_no_further_model_classes():
    assert sorted(class_definitions(PACKAGE_MODELS)) == sorted(SHARED_CLASSES)


@pytest.mark.parametrize('file_name,digest', SAVED_MODELS.items())
def test_the_saved_models_are_unchanged(file_name, digest):
    model_path = os.path.join(os.path.dirname(PACKAGE_MODELS), 'saved_models', file_name)
    file_digest = hashlib.sha256(open(model_path, 'rb').read()).hexdigest()

    assert file_digest.startswith(digest)


@pytest.mark.skipif(not os.path.exists(REPOSITORY_MODELS),
                    reason='the repository models are only available in the repository')
@pytest.mark.parametrize('file_name', SAVED_MODELS)
def test_the_saved_models_are_the_ones_of_the_repository(file_name):
    package_model = os.path.join(os.path.dirname(PACKAGE_MODELS), 'saved_models', file_name)
    repository_model = os.path.join(os.path.dirname(REPOSITORY_MODELS), os.pardir, 'saved_models', file_name)

    assert open(package_model, 'rb').read() == open(repository_model, 'rb').read()
