#ifdef GL_ES
precision highp float;
#endif

uniform vec3 uLightDir;
uniform vec3 uCameraPos;
uniform vec3 uLightRadiance;
uniform sampler2D uGDiffuse;
uniform sampler2D uGDepth;
uniform sampler2D uGNormalWorld;
uniform sampler2D uGShadow;
uniform sampler2D uGPosWorld;

varying mat4 vWorldToScreen;
varying highp vec4 vPosWorld;

#define M_PI 3.1415926535897932384626433832795
#define TWO_PI 6.283185307
#define INV_PI 0.31830988618
#define INV_TWO_PI 0.15915494309

float Rand1(inout float p) {
  p = fract(p * .1031);
  p *= p + 33.33;
  p *= p + p;
  return fract(p);
}

vec2 Rand2(inout float p) {
  return vec2(Rand1(p), Rand1(p));
}

float InitRand(vec2 uv) {
	vec3 p3  = fract(vec3(uv.xyx) * .1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

vec3 SampleHemisphereUniform(inout float s, out float pdf) {
  vec2 uv = Rand2(s);
  float z = uv.x;
  float phi = uv.y * TWO_PI;
  float sinTheta = sqrt(1.0 - z*z);
  vec3 dir = vec3(sinTheta * cos(phi), sinTheta * sin(phi), z);
  pdf = INV_TWO_PI;
  return dir;
}

vec3 SampleHemisphereCos(inout float s, out float pdf) {
  vec2 uv = Rand2(s);
  float z = sqrt(1.0 - uv.x);
  float phi = uv.y * TWO_PI;
  float sinTheta = sqrt(uv.x);
  vec3 dir = vec3(sinTheta * cos(phi), sinTheta * sin(phi), z);
  pdf = z * INV_PI;
  return dir;
}

void LocalBasis(vec3 n, out vec3 b1, out vec3 b2) {
  float sign_ = sign(n.z);
  if (n.z == 0.0) {
    sign_ = 1.0;
  }
  float a = -1.0 / (sign_ + n.z);
  float b = n.x * n.y * a;
  b1 = vec3(1.0 + sign_ * n.x * n.x * a, sign_ * b, -sign_ * n.x);
  b2 = vec3(b, sign_ + n.y * n.y * a, -n.y);
}

vec4 Project(vec4 a) {
  return a / a.w;
}

float GetDepth(vec3 posWorld) {
  float depth = (vWorldToScreen * vec4(posWorld, 1.0)).w;
  return depth;
}

/*
 * Transform point from world space to screen space([0, 1] x [0, 1])
 *
 */
vec2 GetScreenCoordinate(vec3 posWorld) {
  vec2 uv = Project(vWorldToScreen * vec4(posWorld, 1.0)).xy * 0.5 + 0.5;
  return uv;
}

float GetGBufferDepth(vec2 uv) {
  float depth = texture2D(uGDepth, uv).x;
  if (depth < 1e-2) {
    depth = 1000.0;
  }
  return depth;
}

vec3 GetGBufferNormalWorld(vec2 uv) {
  vec3 normal = texture2D(uGNormalWorld, uv).xyz;
  return normal;
}

vec3 GetGBufferPosWorld(vec2 uv) {
  vec3 posWorld = texture2D(uGPosWorld, uv).xyz;
  return posWorld;
}

float GetGBufferuShadow(vec2 uv) {
  float visibility = texture2D(uGShadow, uv).x;
  return visibility;
}

vec3 GetGBufferDiffuse(vec2 uv) {
  vec3 diffuse = texture2D(uGDiffuse, uv).xyz;
  diffuse = pow(diffuse, vec3(1.0/2.2));
  return diffuse;
}

/*
 * Evaluate diffuse bsdf value.
 *
 * wi, wo are all in world space.

 * uv is in screen space, [0, 1] x [0, 1].
 *
 */
vec3 EvalDiffuse(vec3 wi, vec3 wo, vec2 uv) {
  vec3 n = GetGBufferNormalWorld(uv);
  vec3 BSDF  = GetGBufferDiffuse(uv);
  // BSDF /= M_PI;
  return BSDF;
}

/*
 * Evaluate directional light with shadow map
 * uv is in screen space, [0, 1] x [0, 1].
 *
 */
vec3 EvalDirectionalLight(vec2 uv) {
  vec3 n = GetGBufferNormalWorld(uv);
  vec3 wi = normalize(uLightDir);

  float lambort = dot(wi,n);
  if(lambort<0.0){
    lambort =0.0;
  }
  vec3 Le = uLightRadiance*lambort;
  float Vis = GetGBufferuShadow(uv);
  return Le*Vis;
}

vec3 get_step_dir(vec3 dir,int layer){
  return dir*pow(2.,float(layer))*0.01;
}

bool RayMarch(vec3 ori, vec3 dir, out vec3 hitPos) {
  int layer = 0;
  vec3 now_pos = ori;
  vec3 next_pos = now_pos;
  vec3 wo = normalize(uCameraPos - vPosWorld.xyz);
  vec3 n = normalize(GetGBufferNormalWorld(GetScreenCoordinate(ori)));
  float deep_cache = GetDepth(ori);//上次步进深度
  float d_depth = 0.;//深度变化
  for (int i = 0;i<128;i++){
    next_pos = now_pos + get_step_dir(dir,layer);
    float ray_depth = GetDepth(next_pos);
    vec2 uv = GetScreenCoordinate(next_pos);
    if(uv.x<0. || uv.x>1. || uv.y<0. || uv.y>1.){
      if(layer == 0){
        break;
      }else{
        layer--;
        continue;
      }
    }
    float dep = GetGBufferDepth(uv);
    if(dep == 1000.0){
      break;
    }
    if(ray_depth - dep> 1e-2/(max(dot(wo,n),1e-4))){
      if(layer==0){
        hitPos = next_pos;
        return true;
      }else{
        layer--;
        continue;
      }
    }
    d_depth = ray_depth - deep_cache;
    if (d_depth>0.){
      layer++;
    }
    now_pos=next_pos;
    deep_cache = ray_depth;
  }
  // hitPos = vec3(GetGBufferDepth(GetScreenCoordinate(ori+dir)));
  return false;
}

vec3 color_res(vec3 wi,vec3 wo,vec3 pos){
  float s = InitRand(gl_FragCoord.xy);
  vec2 uv = GetScreenCoordinate(pos);
  float dep = GetGBufferDepth(uv);
  vec3 n = GetGBufferNormalWorld(uv);
  vec3 L=vec3(1.0);
  L = EvalDiffuse(wi,wo,uv);
  L *= EvalDirectionalLight(uv);
  return L;
}

#define SAMPLE_NUM 1

void main() {
  vec3 vPosWorld_xyz=vPosWorld.xyz;
  vec2 uv = GetScreenCoordinate(vPosWorld_xyz);
  vec3 n = normalize(GetGBufferNormalWorld(uv));
  vec3 wi = normalize(uLightDir);
  vec3 wo = normalize(uCameraPos - vPosWorld_xyz);
  vec3 hit_pos = vec3(0.0);
  vec3 ref = normalize(reflect(-wo,n));
  vec3 color = vec3(1.0,0.0,1.0);
  if(RayMarch(vPosWorld_xyz,ref,hit_pos)){
     color = color_res(normalize(uLightDir),-ref,hit_pos);
  }else{
    color = vec3(0.0);
  }
  color += color_res(wi,wo,vPosWorld_xyz);
  // if(hit_pos == vec3(0.0)){
  //   color = pow(clamp(L, vec3(0.0), vec3(1.0)), vec3(2.2));
  // }
  // color = pow(clamp(color_res(normal_(uLightDir - hit_pos),-ref,hit_pos), vec3(0.0), vec3(1.0)), vec3(2.2));
  gl_FragColor = vec4(vec3(color.rgb), 1.0);
  // gl_FragColor = vec4(1.0);
  // if(RayMarch(vPosWorld.xyz,ref,hit_pos)){
  //   gl_FragColor = vec4(1.0);
  // }else{
  //   gl_FragColor = vec4(0.0);
  // }
  
  // gl_FragColor = vec4(GetGBufferDiffuse(GetScreenCoordinate(hit_pos)), 1.0);
  // gl_FragColor = vec4(hit_pos, 1.0);
}
