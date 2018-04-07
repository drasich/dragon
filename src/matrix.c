#include "matrix.h"
#include <float.h>
#include <math.h>
#include <stdio.h>

void
mat4_to_mat3(const Matrix4 in, Matrix3 out)
{
  out[0] = in[0];
  out[1] = in[1];
  out[2] = in[2];

  out[3] = in[4];
  out[4] = in[5];
  out[5] = in[6];

  out[6] = in[8];
  out[7] = in[9];
  out[8] = in[10];
}

void
mat3_inverse(const Matrix3 in, Matrix3 out)
{
  //TODO check
  double determinant, inv_determinant;
  double tmp[9];

  tmp[0] = in[4]*in[8] - in[7]*in[5];
  tmp[3] = in[6]*in[5] - in[3]*in[8];
  tmp[6] = in[3]*in[7] - in[6]*in[4];
  tmp[1] = in[7]*in[2] - in[1]*in[8];
  tmp[4] = in[0]*in[8] - in[6]*in[2];
  tmp[7] = in[6]*in[1] - in[0]*in[7];
  tmp[2] = in[1]*in[5] - in[4]*in[2];
  tmp[5] = in[3]*in[2] - in[0]*in[5];
  tmp[8] = in[0]*in[4] - in[3]*in[1];

  determinant =
   in[0]*tmp[0] +
   in[3]*tmp[1] +
   in[6]*tmp[2];

  if ( fabs(determinant) <= DBL_MIN) {
    mat3_set_identity(out);
    return;
   }

  inv_determinant = 1.0f / determinant;
  int i;
  //this was 8 before, I think it was an error so I set it to 9
  printf("mat3 inverse must be checked before use\n)";
  for (i = 0; i < 9; i++){
    out[i] = inv_determinant * tmp[i];
   }
}

void
mat3_set_identity(Matrix3 m)
{
  m[0] = 1;
  m[4] = 1;
  m[8] = 1;
  m[1] = m[2] = m[3] = m[5] = m[6] = m[7] = 0;
}

void
mat4_set_identity(Matrix4 m)
{
  m[0] = 1;
  m[5] = 1;
  m[10] = 1;
  m[15] = 1;
  m[1] = m[2] = m[3] = m[4] = m[6] = m[7] = 0;
  m[8] = m[9] = m[11] = m[12] = m[13] = m[14] = 0;
}

void
mat3_to_gl(const Matrix3 in, Matrix3GL out)
{
  int i;
  for (i = 0; i < 9; ++i) {
    out[i] = (GLfloat) in[i];
  }
}

void
mat4_to_gl(const Matrix4 in, Matrix4GL out)
{
  int i;
  for (i = 0; i < 16; ++i) {
    out[i] = (GLfloat) in[i];
  }
}

void 
mat4_set_frustum(
      Matrix4 m,
      double left,
      double right,
      double bottom,
      double top,
      double near,
      double far)
{
  m[1] = m[3] = m[4] = m[7] = m[2] = m[6] = m[12] = m[13] = 0;

  m[0] = 2 * near / (right - left);
  m[8] = (right + left) / (right - left);
  m[5] = 2 * near / (top - bottom);
  m[9] = (top + bottom) / (top - bottom);
  m[10] = -(far + near) / (far - near);
  m[14] = -(2 * far * near) / (far - near);
  m[11] = -1;
  m[15] = 0;
}

void
mat4_set_perspective(
      Matrix4 m,
      double fovy,
      double aspect,
      double near,
      double far)
{
  double half_height = tan(fovy/2.0)*near;
  double half_width = half_height* aspect;

  mat4_set_frustum(m, -half_width, half_width, -half_height, half_height, near, far);
}

void
mat4_set_orthographic(
      Matrix4 m,
      uint16_t hw,
      uint16_t hh,
      double near,
      double far)
{
  m[1] = m[2] = m[3] = m[4] = m[6] = m[7] = m[8] = m[9] = m[12] = m[13] = m[14] = 0;

  m[0] =  1/(double) hw;
  m[5] =  1/(double) hh;
  m[10] = -2 / (far - near);
  //m[11] = - (far + near) / (far - near);
  m[14] = - (far + near) / (far - near);
  m[15] = 1;
}

void
mat4_multiply(const Matrix4 m, const Matrix4 n, Matrix4 out)
{
  // we use a tmp in case out is the same as m or n.
  // we could make an assert to make sure they are different pointers...
  double tmp[16];

  tmp[0]  = m[0] * n[0]  + m[4] * n[1]  + m[8] * n[2]  + m[12] * n[3];
  tmp[4]  = m[0] * n[4]  + m[4] * n[5]  + m[8] * n[6]  + m[12] * n[7];
  tmp[8]  = m[0] * n[8]  + m[4] * n[9]  + m[8] * n[10] + m[12] * n[11];
  tmp[12] = m[0] * n[12] + m[4] * n[13] + m[8] * n[14] + m[12] * n[15];

  tmp[1]  = m[1] * n[0]  + m[5] * n[1]  + m[9] * n[2]  + m[13] * n[3];
  tmp[5]  = m[1] * n[4]  + m[5] * n[5]  + m[9] * n[6]  + m[13] * n[7];
  tmp[9]  = m[1] * n[8]  + m[5] * n[9]  + m[9] * n[10] + m[13] * n[11];
  tmp[13] = m[1] * n[12] + m[5] * n[13] + m[9] * n[14] + m[13] * n[15];

  tmp[2]  = m[2] * n[0]  + m[6] * n[1]  + m[10] * n[2]  + m[14] * n[3];
  tmp[6]  = m[2] * n[4]  + m[6] * n[5]  + m[10] * n[6]  + m[14] * n[7];
  tmp[10] = m[2] * n[8]  + m[6] * n[9]  + m[10] * n[10] + m[14] * n[11];
  tmp[14] = m[2] * n[12] + m[6] * n[13] + m[10] * n[14] + m[14] * n[15];

  tmp[3]  = m[3] * n[0]  + m[7] * n[1]  + m[11] * n[2]  + m[15] * n[3];
  tmp[7]  = m[3] * n[4]  + m[7] * n[5]  + m[11] * n[6]  + m[15] * n[7];
  tmp[11] = m[3] * n[8]  + m[7] * n[9]  + m[11] * n[10] + m[15] * n[11];
  tmp[15] = m[3] * n[12] + m[7] * n[13] + m[11] * n[14] + m[15] * n[15];

  int i;
  for (i = 0; i < 16; ++i) {
    out[i] = tmp[i];
  }
}

void
mat3_transpose(const Matrix3 in, Matrix3 out)
{
  double tmp[9];

  tmp[0] = in[0];
  tmp[1] = in[3];
  tmp[2] = in[6];

  tmp[3] = in[1];
  tmp[4] = in[4];
  tmp[5] = in[7];

  tmp[6] = in[2];
  tmp[7] = in[5];
  tmp[8] = in[8];

  int i;
  for (i = 0; i < 9; ++i) {
    out[i] = tmp[i];
  }
}


void
mat4_transpose(const Matrix4 in, Matrix4 out)
{
  double tmp[16];

  tmp[0] = in[0];
  tmp[1] = in[4];
  tmp[2] = in[8];
  tmp[3] = in[12];

  tmp[4] = in[1];
  tmp[5] = in[5];
  tmp[6] = in[9];
  tmp[7] = in[13];

  tmp[8] = in[2];
  tmp[9] = in[6];
  tmp[10]= in[10];
  tmp[11]= in[14];

  tmp[12]= in[3];
  tmp[13]= in[7];
  tmp[14]= in[11];
  tmp[15]= in[15];

  int i;
  for (i = 0; i < 16; ++i) {
    out[i] = tmp[i];
  }
}

void
mat4_set_translation(Matrix4 m, Vec3 t)
{
  m[0] = m[5] = m[10] = m[15] = 1;
  m[12] = t.x;
  m[13] = t.y;
  m[14] = t.z;

  m[1] = m[2] = m[4] = 0;
  m[6] = m[8] = m[9] = 0;
  m[3] = m[7] = m[11] = 0;
}

void
mat4_zero(Matrix4 m)
{
  int i;
  for (i = 0; i < 16; ++i) m[i] = 0;
}

void
mat3_zero(Matrix3 m)
{
  int i;
  for (i = 0; i < 9; ++i) m[i] = 0;
}

void 
mat4_set_rotation_quat(Matrix4 m, Quat q)
{
  double length2 = quat_length2(q);
  if (fabs(length2) <= DBL_MIN) {
    mat4_set_identity(m);
    return;
  }

  double rlength2;
  if (length2 != 1) rlength2 = 2.0f/length2;
  else rlength2 = 2;

  double x2, y2, z2, xx, xy, xz, yy, yz, zz, wx, wy, wz;

  x2 = rlength2*q.x;
  y2 = rlength2*q.y;
  z2 = rlength2*q.z;

  xx = q.x * x2;
  xy = q.x * y2;
  xz = q.x * z2;

  yy = q.y * y2;
  yz = q.y * z2;
  zz = q.z * z2;

  wx = q.w * x2;
  wy = q.w * y2;
  wz = q.w * z2;

  m[3] = m[7] = m[11] = m[12] = m[13] = m[14] = 0;

  m[15] = 1;

  m[0] = 1 - (yy + zz);
  m[4] = xy - wz;
  m[8] = xz + wy;

  m[1] = xy + wz;
  m[5] = 1 - (xx + zz);
  m[9] = yz - wx;

  m[2] = xz - wy;
  m[6] = yz + wx;
  m[10] = 1 - (xx + yy);
}

void mat4_set(Matrix4 out,
      double d00,
      double d01,
      double d02,
      double d03,
      double d10,
      double d11,
      double d12,
      double d13,
      double d20,
      double d21,
      double d22,
      double d23,
      double d30,
      double d31,
      double d32,
      double d33
      )
{
  out[0] = d00;
  out[1] = d10;
  out[2] = d20;
  out[3] = d30;

  out[4] = d01; 
  out[5] = d11;
  out[6] = d21;
  out[7] = d31;

  out[8] = d02;
  out[9] = d12;
  out[10]= d22;
  out[11]= d32;

  out[12]= d03; 
  out[13]= d13;
  out[14]= d23;
  out[15]= d33;
}

void
mat4_inverse(const Matrix4 m, Matrix4 out)
{
  //TODO optim
  double m00 = m[0], m01 = m[4], m02 = m[8], m03 = m[12];
  double m10 = m[1], m11 = m[5], m12 = m[9], m13 = m[13];
  double m20 = m[2], m21 = m[6], m22 = m[10], m23 = m[14];
  double m30 = m[3], m31 = m[7], m32 = m[11], m33 = m[15];

  double v0 = m20 * m31 - m21 * m30;
  double v1 = m20 * m32 - m22 * m30;
  double v2 = m20 * m33 - m23 * m30;
  double v3 = m21 * m32 - m22 * m31;
  double v4 = m21 * m33 - m23 * m31;
  double v5 = m22 * m33 - m23 * m32;

  double t00 = + (v5 * m11 - v4 * m12 + v3 * m13);
  double t10 = - (v5 * m10 - v2 * m12 + v1 * m13);
  double t20 = + (v4 * m10 - v2 * m11 + v0 * m13);
  double t30 = - (v3 * m10 - v1 * m11 + v0 * m12);

  double invDet = 1 / (t00 * m00 + t10 * m01 + t20 * m02 + t30 * m03);

  double d00 = t00 * invDet;
  double d10 = t10 * invDet;
  double d20 = t20 * invDet;
  double d30 = t30 * invDet;

  double d01 = - (v5 * m01 - v4 * m02 + v3 * m03) * invDet;
  double d11 = + (v5 * m00 - v2 * m02 + v1 * m03) * invDet;
  double d21 = - (v4 * m00 - v2 * m01 + v0 * m03) * invDet;
  double d31 = + (v3 * m00 - v1 * m01 + v0 * m02) * invDet;

  v0 = m10 * m31 - m11 * m30;
  v1 = m10 * m32 - m12 * m30;
  v2 = m10 * m33 - m13 * m30;
  v3 = m11 * m32 - m12 * m31;
  v4 = m11 * m33 - m13 * m31;
  v5 = m12 * m33 - m13 * m32;

  double d02 = + (v5 * m01 - v4 * m02 + v3 * m03) * invDet;
  double d12 = - (v5 * m00 - v2 * m02 + v1 * m03) * invDet;
  double d22 = + (v4 * m00 - v2 * m01 + v0 * m03) * invDet;
  double d32 = - (v3 * m00 - v1 * m01 + v0 * m02) * invDet;

  v0 = m21 * m10 - m20 * m11;
  v1 = m22 * m10 - m20 * m12;
  v2 = m23 * m10 - m20 * m13;
  v3 = m22 * m11 - m21 * m12;
  v4 = m23 * m11 - m21 * m13;
  v5 = m23 * m12 - m22 * m13;

  double d03 = - (v5 * m01 - v4 * m02 + v3 * m03) * invDet;
  double d13 = + (v5 * m00 - v2 * m02 + v1 * m03) * invDet;
  double d23 = - (v4 * m00 - v2 * m01 + v0 * m03) * invDet;
  double d33 = + (v3 * m00 - v1 * m01 + v0 * m02) * invDet;

  return mat4_set(out,
        d00, d01, d02, d03,
        d10, d11, d12, d13,
        d20, d21, d22, d23,
        d30, d31, d32, d33);
}

Vec3 
mat4_mul(const Matrix4 m, Vec3 v)
{
  return vec3(
        m[0]*v.x + m[4]*v.y + m[8]*v.z,
        m[1]*v.x + m[5]*v.y + m[9]*v.z,
        m[2]*v.x + m[6]*v.y + m[10]*v.z
        );
}

Vec4 
mat4_vec4_mul(const Matrix4 m, Vec4 v)
{
  //TODO check
  return vec4(
        m[0]*v.x + m[4]*v.y + m[8]*v.z + m[12]*v.w,
        m[1]*v.x + m[5]*v.y + m[9]*v.z + m[13]*v.w,
        m[2]*v.x + m[6]*v.y + m[10]*v.z + m[14]*v.w,
        m[3]*v.x + m[7]*v.y + m[11]*v.z + m[15]*v.w
        );
}


Vec3
mat4_premul_unused(const Matrix4 m, Vec3 v)
{
  //TODO and remove unused
  return vec3(
        v.x*m[0] + v.y*m[4] + v.z*m[8],
        v.x*m[1] + v.y*m[5] + v.z*m[9],
        v.x*m[2] + v.y*m[6] + v.z*m[10]);
}

Vec4
mat4_vec4_premul_unused(const Matrix4 m, Vec4 v)
{
  //TODO and remove unused
  return vec4(
        v.x*m[0] + v.y*m[4] + v.z*m[8] + v.w*m[12],
        v.x*m[1] + v.y*m[5] + v.z*m[9] + v.w*m[13],
        v.x*m[2] + v.y*m[6] + v.z*m[10] + v.w*m[14],
        v.x*m[3] + v.y*m[7] + v.z*m[11] + v.w*m[15]
        );
}


void
mat4_pos_ori(Vec3 position, Quat orientation, Matrix4 out)
{
  Matrix4 mt, mr;
  mat4_set_translation(mt, position);
  mat4_set_rotation_quat(mr, orientation);
  mat4_multiply(mt, mr, out);
  //mat4_multiply(mr, mt, out);
}

void 
mat4_lookat(Matrix4 m, Vec3 position, Vec3 at, Vec3 up)
{
  //TODO use quat
  //TODO chris
  Vec3 d = vec3_sub(at, position);
  d = vec3_normalized(d);
  Vec3 s = vec3_cross(d, up);
  s = vec3_normalized(s);
  Vec3 u = vec3_cross(s, d);
  u = vec3_normalized(u);

  m[0] = s.x;
  m[1] = u.x;
  m[2] = -d.x;
  m[3] = 0.0;

  m[4] = s.y;
  m[5] = u.y;
  m[6] = -d.y;
  m[7] = 0.0;

  m[8] = s.z;
  m[9] = u.z;
  m[10] = -d.z;
  m[11] = 0.0;

  m[12] = 0.0;
  m[13] = 0.0;
  m[14] = 0.0;
  m[15] = 1.0;

  //mat4_pre_translate(m, vec3_mul(position, -1));
}

void 
mat4_pre_translate_unused(Matrix4 m, Vec3 v)
{
  //TODO and remove unused, or remove all
  double tmp = v.x;
  if (tmp != 0) {
    m[12] += tmp*m[0];
    m[13] += tmp*m[1];
    m[14] += tmp*m[2];
    m[15] += tmp*m[3];
  }

  tmp = v.y;
  if (tmp != 0) {
    m[12] += tmp*m[4];
    m[13] += tmp*m[5];
    m[14] += tmp*m[6];
    m[15] += tmp*m[7];
  }

  tmp = v.z;
  if (tmp != 0) {
    m[12] += tmp*m[8];
    m[13] += tmp*m[9];
    m[14] += tmp*m[10];
    m[15] += tmp*m[11];
  }
}

typedef struct
{
  Vec4 t;
  Quat q;
  Quat u;
  Quat qk;
  double f;
} _AffineParts;

void 
mat4_decomp_affine(Matrix4 hm, _AffineParts* parts)
{
  //TODO check or remove
  Matrix4 Q, S, U;
  Quat p;

  parts->t = vec4(hm[3], hm[7], hm[11], 0.0);
  //double det = 

}

void 
mat4_decompose_unused(Matrix4 m, Vec3* position, Quat* rotation, Vec3* scale)
{
  //TODO check or remove
  Matrix4 hm;
  mat4_transpose(m, hm);

  _AffineParts parts;
  mat4_decomp_affine(hm, &parts);
}


Quat
mat4_get_quat_sav(Matrix4 m)
{
  //TODO check or remove
  Quat q;

  double s;
  double tq[4];
  int i, j;

  tq[0] = 1 + m[0] + m[5] + m[10];
  tq[1] = 1 + m[0] - m[5] - m[10];
  tq[2] = 1 - m[0] + m[5] - m[10];
  tq[3] = 1 - m[0] - m[5] + m[10];

  j = 0;
  for (i = 1; i < 4; ++i) {
    j = (tq[i] > tq[j]) ? i : j;
  }

  if (j == 0) {
    q.w = tq[0];
    q.x = m[6] - m[9];
    q.y = m[8] - m[2];
    q.z = m[1] - m[4];
  } else if (j == 1) {
    q.w = m[6] - m[9];
    q.x = tq[1];
    q.y = m[1] + m[4];
    q.z = m[8] + m[2];
  } else if (j == 2) {
    q.w = m[8] - m[2];
    q.x = m[1] + m[4];
    q.y = tq[2];
    q.z = m[6] + m[9];
  } else {
    q.w = m[1] - m[4];
    q.x = m[8] + m[2];
    q.y = m[6] + m[9];
    q.z = tq[3];
  }

  s = sqrt(0.25/tq[j]);
  q.w *= s;
  q.x *= s;
  q.y *= s;
  q.z *= s;

  return q;

}


Quat
mat4_get_quat(Matrix4 m)
{
  //TODO check or remove
  Quat q;

  double t = 1 + m[0] + m[5] + m[10];
  double s;

   if (t > 0.00000001) {
     s = sqrt(t) * 2;
     q.x = ( m[9] - m[6] ) / s;
     q.y = ( m[2] - m[8] ) / s;
     q.z = ( m[4] - m[1] ) / s;
     q.w = 0.25 * s;
  } else if (t < 0) {
    printf("c'est plus petit que 0 \n");
  } else {
    if ( m[0] > m[5] && m[0] > m[10] )  {	// Column 0: 
      s  = sqrt( 1.0 + m[0] - m[5] - m[10] ) * 2;
      q.x = 0.25 * s;
      q.y = (m[4] + m[1] ) / s;
      q.z = (m[2] + m[8] ) / s;
      q.w = (m[9] - m[6] ) / s;
    } else if ( m[5] > m[10] ) {			// Column 1: 
      s  = sqrt( 1.0 + m[5] - m[0] - m[10] ) * 2;
      q.x = (m[4] + m[1] ) / s;
      q.y = 0.25 * s;
      q.z = (m[9] + m[6] ) / s;
      q.w = (m[2] - m[8] ) / s;
    } else {						// Column 2:
      s  = sqrt( 1.0 + m[10] - m[0] - m[5] ) * 2;
      q.x = (m[2] + m[8] ) / s;
      q.y = (m[9] + m[6] ) / s;
      q.z = 0.25 * s;
      q.w = (m[4] - m[1] ) / s;
    }
  }

   return q;
}

void
mat4_set_scale(Matrix4 m, const Vec3 v)
{
  m[0] = v.x;
  m[5] = v.y;
  m[10] = v.z;
  m[15] = 1;

  m[1] = m[2] = m[3] = m[4] = 0;
  m[6] = m[7] = m[8] = m[9] = 0;
  m[11] = m[12] = m[13] = m[14] = 0;
}

void
mat4_copy(const Matrix4 in, Matrix4 out)
{
  int i;
  for (i = 0; i < 16; ++i) out[i] = in[i];

}

static void
_mat4_rotate_deg(Matrix4 m, double angle, double x, double y, double z)
{
  double DEG2RAD = M_PI/180;
  double c = cosf(angle * DEG2RAD);
  double s = sinf(angle * DEG2RAD);
  double oneminuscos = 1 - c;

  double xx = x * x;
  double xy = x * y;
  double xz = x * z;
  double yy = y * y;
  double yz = y * z;
  double zz = z * z;

  m[0] = xx * oneminuscos + c;
  m[4] = xy * oneminuscos - z * s;
  m[8] = xz * oneminuscos + y * s;
  m[12] = 0;
  m[1] = xy * oneminuscos + z * s;
  m[5] = yy * oneminuscos + c;
  m[9] = yz * oneminuscos - x * s;
  m[13] = 0;
  m[2] = xz * oneminuscos - y * s;
  m[6] = yz * oneminuscos + x * s;
  m[10]= zz * oneminuscos + c;
  m[14]= 0;
  m[3]= 0;
  m[7]= 0;
  m[11]= 0;
  m[15]= 1;
}

void
mat4_rotation_axis_angle_deg(Matrix4 out, Vec3 axis, double angle)
{
  _mat4_rotate_deg(out, angle, axis.x, axis.y, axis.z);
}

void
mat4_print(const Matrix4 m)
{
  printf("matrix4 : \n");
  int i;
  for (i = 0; i < 4; ++i) {
    printf("  %f %f %f %f\n", m[i], m[i+4], m[i+8], m[i+12]);
  }
}


