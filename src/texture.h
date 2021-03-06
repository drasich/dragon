#ifndef __texture__
#define __texture__
#include <stdbool.h>
#include "fbo.h"
#include "gl.h"
#include "property.h"

enum TextureType
{
  TEX_IMAGE,
  TEX_FBO
};


typedef struct _Texture Texture;

struct _Texture
{
  GLuint  width;
  GLuint  height;

  GLenum  format;
  GLint   internal_format;

  GLubyte *data;
  const char* filename;
  bool is_init;
  int state;

  GLuint id;
};

typedef struct _Id Id;
struct _Id
{
  bool valid;
  GLuint id;
};

Texture* texture_new();

Texture *texture_read_png_file(const char *filename);
bool texture_png_read(Texture* t);
//void save_png(GLfloat* pix);
void save_png(GLuint* pix, int width, int height);

void texture_init(Texture* t);
void texture_clean(Texture* t);

Id* texture_id_get(Texture* t);

#endif
