class DirectionalLight {

    constructor(lightIntensity, lightColor, lightPos, focalPoint, lightUp, hasShadowMap, gl) {
        this.mesh = Mesh.cube(setTransform(0, 0, 0, 0.2, 0.2, 0.2, 0));
        this.mat = new EmissiveMaterial(lightIntensity, lightColor);
        this.lightPos = lightPos;
        this.focalPoint = focalPoint;
        this.lightUp = lightUp

        this.hasShadowMap = hasShadowMap;
        this.fbo = new FBO(gl);
        if (!this.fbo) {
            console.log("无法设置帧缓冲区对象");
            return;
        }
    }
    CalcLightMVP(translate, scale) {
        let lightMVP = mat4.create();
        let modelMatrix = mat4.create();
        let viewMatrix = mat4.create();
        let projectionMatrix = mat4.create();

        // Model transform
        mat4.translate(modelMatrix,modelMatrix,translate);
        mat4.scale(modelMatrix,modelMatrix,scale);
        // View transform
        mat4.lookAt(viewMatrix,this.lightPos,this.focalPoint,this.lightUp);
        mat4.rotateY(viewMatrix,viewMatrix,0.1);
        // Projection transform
        let r = 100.0;
        let l = -r;
        let t = 100.0;
        let b = -t;
        let n = 0.01;
        let f = 200.0;
        mat4.ortho(projectionMatrix,l,r,b,t,n,f);
        mat4.multiply(lightMVP, projectionMatrix, viewMatrix);
        mat4.multiply(lightMVP, lightMVP, modelMatrix);

        return lightMVP;
    }
}
