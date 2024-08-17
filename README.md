# Productionising_ML_Part1
A guide for budding ML Scientists to learn how to write a production level ML code

Notes:

* Adding a MANIFEST.in file can be a good idea, especially if you want to include additional files (like README.md, LICENSE, VERSION, etc.) in your source distribution that are not automatically included by setuptools.

* Always ensure that minmax scaler is the last step as its output is numpy. Pandas based Transformer would fail if applied after it.