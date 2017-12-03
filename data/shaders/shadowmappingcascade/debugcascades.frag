#version 450

layout (binding = 1) uniform sampler2D shadowMap;

layout (location = 0) in vec3 inNormal;
layout (location = 1) in vec3 inColor;
layout (location = 2) in vec3 inViewVec;
layout (location = 3) in vec3 inLightVec;
layout (location = 4) in vec4 inShadowCoord;
layout (location = 5) in vec3 inViewPos;

layout (location = 0) out vec4 outFragColor;

#define SHADOW_MAP_CASCADE_COUNT 4

layout (binding = 2) uniform UBO {
	vec4 cascadeSplits;
} ubo;

void main() 
{	
	vec3 N = normalize(inNormal);
	vec3 L = normalize(inLightVec);
	vec3 V = normalize(inViewVec);
	vec3 R = normalize(-reflect(L, N));
	vec3 diffuse = max(dot(N, L), 0.1) * inColor;

	uint cascadeIndex = 0;
	for(uint i = 0; i < SHADOW_MAP_CASCADE_COUNT - 1; ++i) {
		if(inViewPos.z < ubo.cascadeSplits[i]) {	
			cascadeIndex = i + 1;
		}
	}

	if (cascadeIndex == 0) {
		outFragColor.rgb = diffuse * vec3(1.0f, 0.0f, 0.0f);
	}

	if (cascadeIndex == 1) {
		outFragColor.rgb = diffuse * vec3(0.0f, 1.0f, 0.0f);
	}

	if (cascadeIndex == 2) {
		outFragColor.rgb = diffuse * vec3(0.0f, 0.0f, 1.0f);
	}

	if (cascadeIndex > 2) {
		outFragColor.rgb = diffuse * vec3(1.0f, 1.0f, 0.0f);
	}
}