from IPython.display import Markdown, display
bash = lambda commands: display(Markdown("```bash\n" + ' && \n'.join(commands) + "\n```"))