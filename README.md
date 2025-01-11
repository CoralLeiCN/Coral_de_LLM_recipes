# Coral_de_LLM_recipes

# env setup
uv venv

# activate env
source .venv/bin/activate
# add ipykernel as development dependency
uv add --dev ipykernel
# install ipykernel
uv run ipython kernel install --user --name=llm_recipes