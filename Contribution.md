# 📌 Contributing to DIFFICE-jax

Thank you for considering contributing to `DIFFICE-jax`! We welcome all contributions, whether it’s bug fixes, new features, documentation improvements, or discussions.

---
## How do I contribute

### Reporting issues

- **Check Existing Issues** — Pleae first check whether your issue has already been reported. If yes, directly add comments there.
  
- **Submit a Report** — If you find a new bug, please open a new issue with.
  - A clear description of the problem.
  - Steps to reproduce the issue.
  - Expected behavior.

### Suggesting Improvement

If you Have a great idea to improve our package, please open an issue to discuss your suggestion.

### Adding new datasets

We welcome contributions of new datasets, whether for the existing ice shelves in the examples folder or for new ice shelves. Please ensure that your dataset follows the specified [data format](https://github.com/YaoGroup/DIFFICE_jax/blob/main/docs/data.md). To submit your data, create a Pull Request (see details below).

---
## Pull Request

To contribute via Pull Request, follow these steps:
### Getting Started

1. **Fork the Repository** – Click on the "Fork" button at the top-right corner of this repository.
2. **Clone Your Fork** – Clone your forked repository to your local machine:
   ```sh
   git clone https://github.com/YaoGroup/DIFFICE_jax.git
3. **Navigate to the Project Directory**
   ```sh
   cd DIFFICE_jax
4. **Set Up the Environment** – Install dependencies and ensure everything runs smoothly:
   ```sh
   pip install -r requirements.txt
5. **Install Developer version of DIFFICE_jax**
   ```sh
   pip install -e .
   ```
   An editable version of `DIFFICE_jax` is then installed. All local changes you make to the cloned source code files will be immediately reflected when you import the `diffice_jax` module. Additionly, `pytest` will now be able to locate and test the `diffice_jax` module without any issues.
   

### Contribution Workflow

1. **Create a New Branch**: Always create a feature branch before making changes:
   ```sh
   git checkout -b feature/your-feature-name
2. **Make Your Changes**
   - Ensure your code follows the project's coding style.

3. **Run Unit Test**: we use `pytest` to run unit tests
   ```sh
   python -m pytest  # or pytest
3. **Commit Your Changes**: Write a clear message to describe your changes:
   ```sh
   git commit -m "Add feature: short description"
4. **Push to Your Fork**: Push the branch to your fork on GitHub.
   ```sh
   git push origin feature/your-feature-name
5. **Open a Pull Request (PR)** in the `DIFFICE_jax` repository.
   - Navigate to the main repository.
   - Click on "New Pull Request".
   - Select your branch and describe your changes.

## License

By contributing to pyMechT, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
