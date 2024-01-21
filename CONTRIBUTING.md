# Contributing guidelines

To contribute to the project, you should open a pull request, but before you submit it

1. Make your changes in a new git branch:

   ```shell
   git checkout -b <my-fix/feat-branch> main
   ```

2. Commit your changes using a descriptive commit message that follows
   [commit message conventions](#commit). Adherence to these conventions
   is preferred.

   ```shell
   git commit -a -m '<your commit message>'
   ```

   Note: the optional commit `-a` command line option will automatically "add" and "rm" edited files.

3. Push your branch to GitHub:

   ```shell
   git push origin <my-fix/feat-branch>
   ```

4. In GitHub, send a pull request to `multilayer-perceptron-py:main`.

- If we suggest changes then:

  - Make the required updates.
  - Rebase your branch and force push to your GitHub repository (this will update your Pull Request):

    ```shell
    git rebase main -i
    git push -f
    ```

That's it! Thank you for your contribution!

#### After your pull request is merged

After your pull request is merged, you can safely delete your branch and pull the changes
from the main (upstream) repository:

- Delete the remote branch on GitHub either through the GitHub web UI or your local shell as follows:

  ```shell
  git push origin --delete my-fix-branch
  ```

- Check out the main branch:

  ```shell
  git checkout main -f
  ```

- Delete the local branch:

  ```shell
  git branch -D my-fix-branch
  ```

- Update your main with the latest upstream version:

  ```shell
  git pull --ff upstream main
  ```
