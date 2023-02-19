from ml_cli.commands.evaluate import evaluate
from ml_cli.commands.predict import predict
from ml_cli.commands.train import train
from ml_cli.commands.get_feature_importance import get_feature_importance
import click

@click.group()
def cli():
    pass

cli.add_command(evaluate)
cli.add_command(predict)
cli.add_command(train)
cli.add.command(get_feature_importance)

if __name__ == '__main__':
    cli()
