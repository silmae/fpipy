.. highlight:: shell

============
Contributing
============

Contributions are welcome.

Report bugs or give feedback at https://github.com/silmae/fpipy/issues.

Get Started!
------------

Ready to contribute? Here's how to set up `fpipy` for local development.

1. Fork the `fpipy` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/fpipy.git

3. Install the dependencies using the provided conda environment file
   and then the package itself using pip::

    $ cd fpipy/
    $ conda env create -f envs/development.yml
    $ source activate fpipy-dev
    $ pip install -e .

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass tests::

    $ flake8 fpipy tests
    $ make test

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. Put any new functionality into a function with a docstring.
3. The pull request should work for Python 2.7 and 3.6. Check
   https://travis-ci.org/silmae/fpipy/pull_requests
   and make sure that the tests pass for all supported Python versions.
