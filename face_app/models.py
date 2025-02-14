from django.db import models

class UserFace(models.Model):
    username = models.CharField(max_length=100, unique=True)
    face_encoding = models.TextField()

    def __str__(self):
        return self.username