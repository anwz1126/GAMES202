const LightCubeVertexShader = `
attribute vec3 aVertexPosition;

uniform mat4 uModelViewMatrix;
uniform mat4 uProjectionMatrix;


void main(void) {

  gl_Position = uProjectionMatrix * uModelViewMatrix * vec4(aVertexPosition, 1.0);

}
`;

const LightCubeFragmentShader = `
#ifdef GL_ES
precision mediump float;
#endif

uniform float uLigIntensity;
uniform vec3 uLightColor;

void main(void) {
    
  //gl_FragColor = vec4(1,1,1, 1.0);
  gl_FragColor = vec4(uLightColor, 1.0);
}
`;
const VertexShader = `
attribute vec3 aVertexPosition;
attribute vec3 aNormalPosition;
attribute vec2 aTextureCoord;

uniform mat4 uModelViewMatrix;
uniform mat4 uProjectionMatrix;

varying highp vec3 vFragPos;
varying highp vec3 vNormal;
varying highp vec2 vTextureCoord;

void main(void) {

  vFragPos = aVertexPosition;
  vNormal = aNormalPosition;

  gl_Position = uProjectionMatrix * uModelViewMatrix * vec4(aVertexPosition, 1.0);

  vTextureCoord = aTextureCoord;

}
`;

const FragmentShader = `
#ifdef GL_ES
precision mediump float;
#endif

uniform int uTextureSample;
uniform vec3 uKd;
uniform sampler2D uSampler;
uniform vec3 uLightPos;
uniform vec3 uCameraPos;

varying highp vec3 vFragPos;
varying highp vec3 vNormal;
varying highp vec2 vTextureCoord;

void main(void) {
  
  if (uTextureSample == 1) {
    gl_FragColor = texture2D(uSampler, vTextureCoord);
  } else {
    gl_FragColor = vec4(uKd,1);
  }

}
`;

const PhongVertexShader = `
attribute vec3 aVertexPosition;
attribute vec3 aNormalPosition;
attribute vec2 aTextureCoord;

uniform mat4 uModelViewMatrix;
uniform mat4 uProjectionMatrix;

varying highp vec2 vTextureCoord;
varying highp vec3 vFragPos;
varying highp vec3 vNormal;

void main(void) {

vFragPos = aVertexPosition;
vNormal = aNormalPosition;

gl_Position = uProjectionMatrix * uModelViewMatrix * vec4(aVertexPosition , 1.0);

vTextureCoord = aTextureCoord;
}

`;

const PhongFragmentShader = `
#ifdef GL_ES
precision mediump float;
#endif
uniform sampler2D uSampler;
//binn
uniform vec3 uKd;
uniform vec3 uKs;
uniform vec3 uLightPos;
uniform vec3 uCameraPos;
uniform float uLightIntensity;
uniform int uTextureSample;

varying highp vec2 vTextureCoord;
varying highp vec3 vFragPos;
varying highp vec3 vNormal;

void main(void) {
vec3 color;
if (uTextureSample == 1) {
color = pow(texture2D(uSampler , vTextureCoord).rgb, vec3(1));
} else {
color = uKd;
}

vec3 ambient = 0.05 * color;

vec3 lightDir = normalize(uLightPos - vFragPos);
vec3 normal = normalize(vNormal);
//float diff = max(dot(lightDir , normal), 0.0);
float diff = dot(normalize(lightDir*vec3(1.0,0.0,1.0)) , normal);
if(diff>.998){
  diff = 1.;
}else if(diff>-.7){
  diff = 0.93;
}else{
  diff = 0.85;
}
float light_atten_coff = uLightIntensity / length(uLightPos - vFragPos);
//vec3 diffuse = diff * light_atten_coff * color;
vec3 diffuse = diff * color;

vec3 viewDir = normalize(uCameraPos - vFragPos);
float spec = 0.0;
vec3 reflectDir = reflect(-lightDir , normal);
spec = pow (max(dot(viewDir , reflectDir), 0.0), 35.0);
vec3 specular = uKs * light_atten_coff * spec;

float bianyuan = (1.-max(dot(normalize(vNormal),viewDir),0.0)) * max(dot(normalize(vNormal),lightDir*vec3(1.0,0.0,1.0)),0.0);
bianyuan = max(pow(bianyuan,5.0),0.0);

gl_FragColor = vec4(pow((ambient + diffuse + specular + bianyuan * light_atten_coff), vec3(2.2)), 1.0);
//gl_FragColor = vec4(pow((bianyuan * vec3(1.)), vec3(2.2)), 1.0);
}
`;