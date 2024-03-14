#ifdef GL_ES
precision mediump float;
#endif

// Phong related variables
uniform sampler2D uSampler;
uniform vec3 uKd;
uniform vec3 uKs;
uniform vec3 uLightPos;
uniform vec3 uCameraPos;
uniform vec3 uLightIntensity;

varying highp vec2 vTextureCoord;
varying highp vec3 vFragPos;
varying highp vec3 vNormal;

// Shadow map related variables
#define NUM_SAMPLES 20
#define BLOCKER_SEARCH_NUM_SAMPLES NUM_SAMPLES
#define PCF_NUM_SAMPLES NUM_SAMPLES
#define NUM_RINGS 10

#define EPS 1e-3
#define PI 3.141592653589793
#define PI2 6.283185307179586

#define SHADOW_MAP_SIZE 2048.
#define FRUSTUM_SIZE  400.
#define light_rangesize 0.01

uniform sampler2D uShadowMap;

varying vec4 vPositionFromLight;

highp float rand_1to1(highp float x ) { 
  // -1 -1
  return fract(sin(x)*10000.0);
}

highp float rand_2to1(vec2 uv ) { 
  // 0 - 1
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract(sin(sn) * c);
}

float unpack(vec4 rgbaDepth) {
    const vec4 bitShift = vec4(1.0, 1.0/256.0, 1.0/(256.0*256.0), 1.0/(256.0*256.0*256.0));
    return dot(rgbaDepth, bitShift);
}

vec2 poissonDisk[NUM_SAMPLES];

void poissonDiskSamples( const in vec2 randomSeed ) {

  float ANGLE_STEP = PI2 * float( NUM_RINGS ) / float( NUM_SAMPLES );
  float INV_NUM_SAMPLES = 1.0 / float( NUM_SAMPLES );

  float angle = rand_2to1( randomSeed ) * PI2;
  float radius = INV_NUM_SAMPLES;
  float radiusStep = radius;

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( cos( angle ), sin( angle ) ) * pow( radius, 0.75 );
    radius += radiusStep;
    angle += ANGLE_STEP;
  }
}

void uniformDiskSamples( const in vec2 randomSeed ) {

  float randNum = rand_2to1(randomSeed);
  float sampleX = rand_1to1( randNum ) ;
  float sampleY = rand_1to1( sampleX ) ;

  float angle = sampleX * PI2;
  float radius = sqrt(sampleY);

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( radius * cos(angle) , radius * sin(angle)  );

    sampleX = rand_1to1( sampleY ) ;
    sampleY = rand_1to1( sampleX ) ;

    angle = sampleX * PI2;
    radius = sqrt(sampleY);
  }
}

float findBlocker( sampler2D shadowMap,  vec2 uv, float zReceiver ) {
  uniformDiskSamples(uv);
  int blockerNum=0;
  float blockerSum=0.;

  vec3 normal = normalize(vNormal);
  vec3 lightDir = normalize(uLightPos - vFragPos);
  float bias = (1.0 - dot(normal, lightDir))*(FRUSTUM_SIZE / SHADOW_MAP_SIZE / 2.);
  
  for(int i = 0;i<NUM_SAMPLES;i++){
    // float shadow_depth=unpack(texture2D(shadowMap,uv+poissonDisk[i]/SHADOW_MAP_SIZE*light_rangesize));
    float shadow_depth=unpack(texture2D(shadowMap,uv+poissonDisk[i]*light_rangesize));
    if(shadow_depth<=zReceiver-EPS-bias*1.*pow(zReceiver,1.0)-.1){
      blockerNum++;
      blockerSum+=shadow_depth;
    }
  }
  if(blockerNum==0){
    return zReceiver;
  }
  //light_rangesize
  // return blockerSum/float(NUM_SAMPLES);
  return blockerSum/float(blockerNum);
}

float useShadowMap(sampler2D shadowMap, vec4 shadowCoord){
  // if(shadowCoord.x>1. || shadowCoord.x<0. || shadowCoord.y>1. || shadowCoord.y<0.){
  //   return 1.;
  // }
  vec3 normal = normalize(vNormal);
  vec3 lightDir = normalize(uLightPos - vFragPos);
  float bias = (1.0 - dot(normal, lightDir))*(FRUSTUM_SIZE / SHADOW_MAP_SIZE / 2.);
  //return unpack(texture2D(shadowMap,shadowCoord.xy));
  if(unpack(texture2D(shadowMap,shadowCoord.xy))<=shadowCoord.z-EPS-bias*.1*pow(shadowCoord.z,1.0)-.01){
    return .0;
  }
  return 1.;
}

float PCF(sampler2D shadowMap, vec4 coords) {
  uniformDiskSamples(coords.xy);
  //poissonDiskSamples(coords.xy);
  float v=0.;
  for(int i = 0;i<NUM_SAMPLES;i++){
    v+=useShadowMap(shadowMap,coords+vec4(poissonDisk[i].xy/SHADOW_MAP_SIZE*16.,0.,0.));
  }
  return v/float(NUM_SAMPLES);
}

float PCSS(sampler2D shadowMap, vec4 coords){
  // if(coords.x>1. || coords.x<0. || coords.y>1. || coords.y<0.){
  //   return 1.;
  // }
  // float light_phy_zise = 1.;//meter
  float receiver_deepth = coords.z;
  // STEP 1: avgblocker depth
  float blocker_av_depth = findBlocker(shadowMap,coords.xy,receiver_deepth);

  // STEP 2: penumbra size
  float penumbra_size = (receiver_deepth-blocker_av_depth)*(light_rangesize*SHADOW_MAP_SIZE)/blocker_av_depth;
  // STEP 3: filtering
  //(PCF)
  float v=0.;
  uniformDiskSamples(coords.xy);
  for(int i = 0;i<NUM_SAMPLES;i++){
    v+=useShadowMap(shadowMap,coords+vec4(poissonDisk[i].xy/SHADOW_MAP_SIZE*penumbra_size,0.,0.));
  }
  // return blocker_av_depth;
  return v/float(NUM_SAMPLES);

}



vec3 blinnPhong() {
  vec3 color = texture2D(uSampler, vTextureCoord).rgb;
  //color = pow(color, vec3(1.));

  vec3 ambient = 0.05 * color;

  vec3 lightDir = normalize(uLightPos);
  vec3 normal = normalize(vNormal);
  float diff = max(dot(lightDir, normal), 0.0);


  vec3 light_atten_coff =
      uLightIntensity / pow(length(uLightPos - vFragPos), 2.0);
  vec3 diffuse = diff * light_atten_coff * color;

  vec3 viewDir = normalize(uCameraPos - vFragPos);
  vec3 halfDir = normalize((lightDir + viewDir));
  float spec = pow(max(dot(halfDir, normal), 0.0), 32.0);
  vec3 specular = uKs * light_atten_coff * spec;

  vec3 radiance = (ambient + diffuse + specular);
  vec3 phongColor = pow(radiance, vec3(1.));
  return phongColor;
}

vec3 toon3to2(float visibility){
  vec3 color = texture2D(uSampler, vTextureCoord).rgb;
  float k_ambient = 0.65;
  vec3 lightDir = normalize(uLightPos);
  vec3 normal = normalize(vNormal);
  float lambort = max(dot(lightDir, normal), 0.0);
  vec3 light_atten_coff =
      uLightIntensity / pow(length(uLightPos - vFragPos), 2.0);
  if(lambort>.995){
    lambort = .3;//spec
  }else if(lambort>.1){
    lambort = .28;//deff
  }else{
    lambort = .13;//shadow
  }
  return k_ambient* color + visibility * lambort * color*light_atten_coff;
}
void main(void) {
  //vFragPos
  vec3 shadowCoord = vPositionFromLight.xyz/vPositionFromLight.w*.5+vec3(.5,.5,.5);
  float visibility;
  // visibility = useShadowMap(uShadowMap, vec4(shadowCoord, 1.0));
  // visibility = PCF(uShadowMap, vec4(shadowCoord, 1.0));
  visibility = PCSS(uShadowMap, vec4(shadowCoord, 1.0));

  vec3 phongColor = blinnPhong();
  vec3 toon3to2 = toon3to2(visibility);

  // gl_FragColor = vec4(phongColor * visibility, 1.0);
  gl_FragColor = vec4(pow(vec3(toon3to2),vec3(2.2)), 1.0);
  //gl_FragColor = vec4(pow(vec3(vTextureCoord,0.),vec3(2.2)),0.);
  // gl_FragColor=vec4(visibility);
}