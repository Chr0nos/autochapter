from tortoise.models import Model
from tortoise import fields
from tortoise_vector.field import VectorField


class File(Model):
    id = fields.IntField(pk=True)
    filename = fields.CharField(max_length=1000, index=True)
    scanned_at = fields.DatetimeField(auto_now_add=True)
    fps = fields.IntField()

    def __str__(self) -> str:
        return self.filename


class Frame(Model):
    id = fields.IntField(pk=True)
    file = fields.ForeignKeyField('main.File', related_name='frames', on_delete=fields.CASCADE)
    offset = fields.TimeDeltaField()
    embedding = VectorField(vector_size=512)
