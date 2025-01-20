# Coral_de_LLM_recipes

# env setup
uv venv

# activate env
source .venv/bin/activate

# Install the locked requirements:
uv pip sync requirements.txt

# add ipykernel as development dependency
uv add --dev ipykernel
# install ipykernel
uv run ipython kernel install --user --name=llm_recipes

# Checking if the lockfile is up-to-date
uv lock --check

# update lock file
uv lock