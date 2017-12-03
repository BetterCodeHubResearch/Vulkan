/*
* Vulkan Example - Cascaded shadow mapping for directional light sources
*
* Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

// Note: Could be simplified with a layered frame buffer using geometry shaders (not available on all devices)

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>


#include <vulkan/vulkan.h>
#include "vulkanexamplebase.h"
#include "VulkanBuffer.hpp"
#include "VulkanModel.hpp"

#define VERTEX_BUFFER_BIND_ID 0
#define ENABLE_VALIDATION false

// 16 bits of depth is enough for such a small scene
#define DEPTH_FORMAT VK_FORMAT_D16_UNORM

// Shadowmap properties
#if defined(__ANDROID__)
#define SHADOWMAP_DIM 1024
#else
#define SHADOWMAP_DIM 2048
#endif
#define SHADOWMAP_FILTER VK_FILTER_LINEAR

#define SHADOW_MAP_CASCADE_COUNT 4
#define SPLIT_LOG_FACTOR 0.9f

// Offscreen frame buffer properties
#define FB_COLOR_FORMAT VK_FORMAT_R8G8B8A8_UNORM

class VulkanExample : public VulkanExampleBase
{
public:
	bool displayShadowMap = false;
	int32_t displayShadowMapCascadeIndex = 0;
	bool displayCascades = false;
	bool filterPCF = true;

	float cascadeSplitLambda = 1.0f;

	float zNear = 1.0f;
	float zFar = 48.0f;

	// Depth bias (and slope) are used to avoid shadowing artefacts
	// Constant depth bias factor (always applied)
	float depthBiasConstant = 1.25f;
	// Slope depth bias factor, applied depending on polygon's slope
	float depthBiasSlope = 1.75f;

	glm::vec3 lightPos = glm::vec3();
	float lightFOV = 45.0f;

	// Vertex layout for the models
	vks::VertexLayout vertexLayout = vks::VertexLayout({
		vks::VERTEX_COMPONENT_POSITION,
		vks::VERTEX_COMPONENT_UV,
		vks::VERTEX_COMPONENT_COLOR,
		vks::VERTEX_COMPONENT_NORMAL,
	});

	std::vector<vks::Model> scenes;
	std::vector<std::string> sceneNames;
	int32_t sceneIndex = 0;

	struct {
		vks::Buffer scene;
		vks::Buffer sceneFS;
		vks::Buffer offscreen;
	} uniformBuffers;

	struct {
		glm::mat4 projection;
		glm::mat4 model;
	} uboVSquad;

	struct {
		glm::mat4 projection;
		glm::mat4 view;
		glm::mat4 model;
		glm::mat4 depthBiasMVP;
		glm::vec3 lightPos;
	} uboVSscene;

	struct {
		float cascadeSplits[4];
	} uboFSscene;

	struct {
		std::array<glm::mat4, SHADOW_MAP_CASCADE_COUNT> depthMVP;
	} uboOffscreenVS;

	struct {
		VkPipeline debugShadowMap;
		VkPipeline debugCascades;
		VkPipeline shadowMap;
		VkPipeline sceneShadow;
		VkPipeline sceneShadowPCF;
	} pipelines;

	struct {
		VkPipelineLayout shared;
		VkPipelineLayout offscreen;
	} pipelineLayouts;

	VkDescriptorSet descriptorSet;
	VkDescriptorSetLayout descriptorSetLayout;

	struct ShadowPass {
		VkRenderPass renderPass;
		VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
		VkSemaphore semaphore = VK_NULL_HANDLE;
	} shadowPass;

	// Contains all resources required for a single shadow map cascade
	struct Cascade {
		VkFramebuffer frameBuffer;
		struct DepthImage {
			VkImage image;
			VkDeviceMemory mem;
			VkImageView view;
		} depth;
		VkDescriptorSet descriptorSet;

		float splitDepth;
		glm::mat4 viewProjMatrix;

		void destroy(VkDevice device) {
			vkDestroyImageView(device, depth.view, nullptr);
			vkDestroyImage(device, depth.image, nullptr);
			vkFreeMemory(device, depth.mem, nullptr);
			vkDestroyFramebuffer(device, frameBuffer, nullptr);
		}
	};
	std::array<Cascade, SHADOW_MAP_CASCADE_COUNT> cascades;

	// All cascades share the same texture sampler
	VkSampler cascadeSampler;

	std::array<float, SHADOW_MAP_CASCADE_COUNT> splitDepths;

	VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
	{
		title = "Cascaded shadow mapping";
		timerSpeed *= 0.5f;
		camera.type = Camera::CameraType::firstperson;
		camera.movementSpeed = 7.5f;
		camera.setPosition(glm::vec3(5.6f, 3.6f, 5.3f));
		camera.setRotation(glm::vec3(-36.0f, 135.0f, 0.0f));
		camera.setPerspective(45.0f, (float)width / (float)height, zNear, zFar);
		settings.overlay = true;
	}

	~VulkanExample()
	{
		// Clean up used Vulkan resources 
		// Note : Inherited destructor cleans up resources stored in base class

		vkDestroySampler(device, cascadeSampler, nullptr);

		for (auto cascade : cascades) {
			cascade.destroy(device);
		}

		vkDestroyRenderPass(device, shadowPass.renderPass, nullptr);

		vkDestroyPipeline(device, pipelines.debugShadowMap, nullptr);
		vkDestroyPipeline(device, pipelines.shadowMap, nullptr);
		vkDestroyPipeline(device, pipelines.sceneShadow, nullptr);
		vkDestroyPipeline(device, pipelines.sceneShadowPCF, nullptr);

		vkDestroyPipelineLayout(device, pipelineLayouts.shared, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayouts.offscreen, nullptr);

		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

		for (auto scene : scenes) {
			scene.destroy();
		}

		uniformBuffers.offscreen.destroy();
		uniformBuffers.scene.destroy();

		vkFreeCommandBuffers(device, cmdPool, 1, &shadowPass.commandBuffer);
		vkDestroySemaphore(device, shadowPass.semaphore, nullptr);
	}

	// Set up a separate render pass for the offscreen frame buffer
	// This is necessary as the offscreen frame buffer attachments use formats different to those from the example render pass
	void prepareOffscreenRenderpass()
	{
		VkAttachmentDescription attachmentDescription{};
		attachmentDescription.format = DEPTH_FORMAT;
		attachmentDescription.samples = VK_SAMPLE_COUNT_1_BIT;
		attachmentDescription.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;							// Clear depth at beginning of the render pass
		attachmentDescription.storeOp = VK_ATTACHMENT_STORE_OP_STORE;						// We will read from depth, so it's important to store the depth attachment results
		attachmentDescription.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		attachmentDescription.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachmentDescription.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;					// We don't care about initial layout of the attachment
		attachmentDescription.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;// Attachment will be transitioned to shader read at render pass end

		VkAttachmentReference depthReference = {};
		depthReference.attachment = 0;
		depthReference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;			// Attachment will be used as depth/stencil during render pass

		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 0;													// No color attachments
		subpass.pDepthStencilAttachment = &depthReference;									// Reference to our depth attachment

		// Use subpass dependencies for layout transitions
		std::array<VkSubpassDependency, 2> dependencies;

		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		dependencies[1].srcSubpass = 0;
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		VkRenderPassCreateInfo renderPassCreateInfo = vks::initializers::renderPassCreateInfo();
		renderPassCreateInfo.attachmentCount = 1;
		renderPassCreateInfo.pAttachments = &attachmentDescription;
		renderPassCreateInfo.subpassCount = 1;
		renderPassCreateInfo.pSubpasses = &subpass;
		renderPassCreateInfo.dependencyCount = static_cast<uint32_t>(dependencies.size());
		renderPassCreateInfo.pDependencies = dependencies.data();

		VK_CHECK_RESULT(vkCreateRenderPass(device, &renderPassCreateInfo, nullptr, &shadowPass.renderPass));
	}

	// Setup the offscreen framebuffer for rendering the scene from light's point-of-view to
	// The depth attachment of this framebuffer will then be used to sample from in the fragment shader of the shadowing pass
	void prepareOffscreenFramebuffer()
	{
		prepareOffscreenRenderpass();

		VkFormat fbColorFormat = FB_COLOR_FORMAT;

		// One image and framebuffer per cascade
		for (uint32_t i = 0; i < SHADOW_MAP_CASCADE_COUNT; i++) {

			// Depth image 
			VkImageCreateInfo imageInfo = vks::initializers::imageCreateInfo();
			imageInfo.imageType = VK_IMAGE_TYPE_2D;
			imageInfo.extent.width = SHADOWMAP_DIM;
			imageInfo.extent.height = SHADOWMAP_DIM;
			imageInfo.extent.depth = 1;
			imageInfo.mipLevels = 1;
			imageInfo.arrayLayers = 1;
			imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
			imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
			imageInfo.format = DEPTH_FORMAT;
			imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
			VK_CHECK_RESULT(vkCreateImage(device, &imageInfo, nullptr, &cascades[i].depth.image));

			VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
			VkMemoryRequirements memReqs;
			vkGetImageMemoryRequirements(device, cascades[i].depth.image, &memReqs);
			memAlloc.allocationSize = memReqs.size;
			memAlloc.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
			VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, &cascades[i].depth.mem));
			VK_CHECK_RESULT(vkBindImageMemory(device, cascades[i].depth.image, cascades[i].depth.mem, 0));

			// Image view
			VkImageViewCreateInfo viewInfo = vks::initializers::imageViewCreateInfo();
			viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
			viewInfo.format = DEPTH_FORMAT;
			viewInfo.subresourceRange = {};
			viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
			viewInfo.subresourceRange.baseMipLevel = 0;
			viewInfo.subresourceRange.levelCount = 1;
			viewInfo.subresourceRange.baseArrayLayer = 0;
			viewInfo.subresourceRange.layerCount = 1;
			viewInfo.image = cascades[i].depth.image;
			VK_CHECK_RESULT(vkCreateImageView(device, &viewInfo, nullptr, &cascades[i].depth.view));

			// Framebuffer
			VkFramebufferCreateInfo framebufferInfo = vks::initializers::framebufferCreateInfo();
			framebufferInfo.renderPass = shadowPass.renderPass;
			framebufferInfo.attachmentCount = 1;
			framebufferInfo.pAttachments = &cascades[i].depth.view;
			framebufferInfo.width = SHADOWMAP_DIM;
			framebufferInfo.height = SHADOWMAP_DIM;
			framebufferInfo.layers = 1;
			VK_CHECK_RESULT(vkCreateFramebuffer(device, &framebufferInfo, nullptr, &cascades[i].frameBuffer));
		}

		// Shared sampler for depth reads
		VkSamplerCreateInfo sampler = vks::initializers::samplerCreateInfo();
		sampler.magFilter = SHADOWMAP_FILTER;
		sampler.minFilter = SHADOWMAP_FILTER;
		sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
		sampler.addressModeV = sampler.addressModeU;
		sampler.addressModeW = sampler.addressModeU;
		sampler.mipLodBias = 0.0f;
		sampler.maxAnisotropy = 1.0f;
		sampler.minLod = 0.0f;
		sampler.maxLod = 1.0f;
		sampler.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
		VK_CHECK_RESULT(vkCreateSampler(device, &sampler, nullptr, &cascadeSampler));
	}

	void buildOffscreenCommandBuffer()
	{
		if (shadowPass.commandBuffer == VK_NULL_HANDLE)
		{
			shadowPass.commandBuffer = VulkanExampleBase::createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, false);
		}
		if (shadowPass.semaphore == VK_NULL_HANDLE)
		{
			// Create a semaphore used to synchronize offscreen rendering and usage
			VkSemaphoreCreateInfo semaphoreCreateInfo = vks::initializers::semaphoreCreateInfo();
			VK_CHECK_RESULT(vkCreateSemaphore(device, &semaphoreCreateInfo, nullptr, &shadowPass.semaphore));
		}

		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VkClearValue clearValues[1];
		clearValues[0].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = shadowPass.renderPass;
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.renderArea.extent.width = SHADOWMAP_DIM;
		renderPassBeginInfo.renderArea.extent.height = SHADOWMAP_DIM;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		VK_CHECK_RESULT(vkBeginCommandBuffer(shadowPass.commandBuffer, &cmdBufInfo));

		VkViewport viewport = vks::initializers::viewport((float)SHADOWMAP_DIM, (float)SHADOWMAP_DIM, 0.0f, 1.0f);
		vkCmdSetViewport(shadowPass.commandBuffer, 0, 1, &viewport);

		VkRect2D scissor = vks::initializers::rect2D(SHADOWMAP_DIM, SHADOWMAP_DIM, 0, 0);
		vkCmdSetScissor(shadowPass.commandBuffer, 0, 1, &scissor);

		// Multi-pass shadow map cascade generation
		// Could be simplified to one pass using geometry shaders and layered frame buffers
		for (uint32_t i = 0; i < SHADOW_MAP_CASCADE_COUNT; i++) {
			renderPassBeginInfo.framebuffer = cascades[i].frameBuffer;
			vkCmdBeginRenderPass(shadowPass.commandBuffer, &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);
			vkCmdBindPipeline(shadowPass.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.shadowMap);
			vkCmdPushConstants(shadowPass.commandBuffer, pipelineLayouts.offscreen, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(uint32_t), &i);
			VkDeviceSize offsets[1] = { 0 };
			vkCmdBindDescriptorSets(shadowPass.commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.offscreen, 0, 1, &cascades[i].descriptorSet, 0, NULL);
			vkCmdBindVertexBuffers(shadowPass.commandBuffer, VERTEX_BUFFER_BIND_ID, 1, &scenes[sceneIndex].vertices.buffer, offsets);
			vkCmdBindIndexBuffer(shadowPass.commandBuffer, scenes[sceneIndex].indices.buffer, 0, VK_INDEX_TYPE_UINT32);
			vkCmdDrawIndexed(shadowPass.commandBuffer, scenes[sceneIndex].indexCount, 1, 0, 0, 0);
			vkCmdEndRenderPass(shadowPass.commandBuffer);
		}

		VK_CHECK_RESULT(vkEndCommandBuffer(shadowPass.commandBuffer));
	}

	void buildCommandBuffers()
	{
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		VkClearValue clearValues[2];
		clearValues[0].color = defaultClearColor;
		clearValues[1].depthStencil = { 1.0f, 0 };

		VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
		renderPassBeginInfo.renderPass = renderPass;
		renderPassBeginInfo.renderArea.offset.x = 0;
		renderPassBeginInfo.renderArea.offset.y = 0;
		renderPassBeginInfo.renderArea.extent.width = width;
		renderPassBeginInfo.renderArea.extent.height = height;
		renderPassBeginInfo.clearValueCount = 2;
		renderPassBeginInfo.pClearValues = clearValues;

		for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
		{
			// Set target frame buffer
			renderPassBeginInfo.framebuffer = frameBuffers[i];

			VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

			vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

			VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
			vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

			VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
			vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

			VkDeviceSize offsets[1] = { 0 };

			// Visualize shadow map cascade
			if (displayShadowMap) {
				vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.shared, 0, 1, &cascades[displayShadowMapCascadeIndex].descriptorSet, 0, NULL);
				vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.debugShadowMap);
				vkCmdDraw(drawCmdBuffers[i], 3, 1, 0, 0);
			}

			// Render shadowed scene
			vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayouts.shared, 0, 1, &descriptorSet, 0, NULL);

			if (displayCascades) {
				vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelines.debugCascades);
			}
			else {
				vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, (filterPCF) ? pipelines.sceneShadowPCF : pipelines.sceneShadow);
			}

			vkCmdBindVertexBuffers(drawCmdBuffers[i], VERTEX_BUFFER_BIND_ID, 1, &scenes[sceneIndex].vertices.buffer, offsets);
			vkCmdBindIndexBuffer(drawCmdBuffers[i], scenes[sceneIndex].indices.buffer, 0, VK_INDEX_TYPE_UINT32);
			vkCmdDrawIndexed(drawCmdBuffers[i], scenes[sceneIndex].indexCount, 1, 0, 0, 0);

			vkCmdEndRenderPass(drawCmdBuffers[i]);

			VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
		}
	}

	void loadAssets()
	{
		scenes.resize(2);
		scenes[0].loadFromFile(getAssetPath() + "models/terrain.obj", vertexLayout, 2.0f, vulkanDevice, queue);
		scenes[1].loadFromFile(getAssetPath() + "models/samplescene.dae", vertexLayout, 0.25f, vulkanDevice, queue);
		sceneNames = {"Terrain test", "Teapots and pillars" };
	}

	void setupDescriptorPool()
	{
		// Example uses three ubos and two image samplers
		std::vector<VkDescriptorPoolSize> poolSizes =
		{
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10),
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10)
		};

		VkDescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(
				poolSizes.size(),
				poolSizes.data(),
				3 + SHADOW_MAP_CASCADE_COUNT);

		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));
	}

	void setupDescriptorSetLayout()
	{
		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT, 1),
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT, 2),
		};

		VkDescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout));

		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts.shared));

		// Offscreen pipeline layout
		// Pass cascade index as push constant
		VkPushConstantRange pushConstantRange =
			vks::initializers::pushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, sizeof(uint32_t), 0);
		pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
		pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayouts.offscreen));
	}

	void setupDescriptorSets()
	{
		std::vector<VkWriteDescriptorSet> writeDescriptorSets;

		VkDescriptorSetAllocateInfo allocInfo =
			vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);

		// Scene rendering / debug display
		VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));
		// Image descriptor for the shadow map attachment
		// todo: pass cascade array
		VkDescriptorImageInfo texDescriptor =
			vks::initializers::descriptorImageInfo(
				cascadeSampler,
				cascades[0].depth.view,
				VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
		writeDescriptorSets = {
			vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformBuffers.scene.descriptor),
			vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &texDescriptor),
			vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2, &uniformBuffers.sceneFS.descriptor),
		};
		vkUpdateDescriptorSets(device, writeDescriptorSets.size(), writeDescriptorSets.data(), 0, NULL);

		// Per-cascade descriptor set
		// Each descriptor set represents a single layer of the array texture
		for (uint32_t i = 0; i < SHADOW_MAP_CASCADE_COUNT; i++) {
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &cascades[i].descriptorSet));
			VkDescriptorImageInfo cascadeImageInfo = vks::initializers::descriptorImageInfo(
				cascadeSampler,
				cascades[i].depth.view,
				VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL);
			writeDescriptorSets = {
				vks::initializers::writeDescriptorSet(cascades[i].descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformBuffers.offscreen.descriptor),
				vks::initializers::writeDescriptorSet(cascades[i].descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, &cascadeImageInfo)
			};
			vkUpdateDescriptorSets(device, writeDescriptorSets.size(), writeDescriptorSets.data(), 0, NULL);
		}
	}

	void preparePipelines()
	{
		VkPipelineInputAssemblyStateCreateInfo inputAssemblyState =
			vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0, VK_FALSE);

		VkPipelineRasterizationStateCreateInfo rasterizationState =
			vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_BACK_BIT, VK_FRONT_FACE_CLOCKWISE, 0);

		VkPipelineColorBlendAttachmentState blendAttachmentState =
			vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);

		VkPipelineColorBlendStateCreateInfo colorBlendState =
			vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);

		VkPipelineDepthStencilStateCreateInfo depthStencilState =
			vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);

		VkPipelineViewportStateCreateInfo viewportState =
			vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);

		VkPipelineMultisampleStateCreateInfo multisampleState =
			vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT, 0);

		std::vector<VkDynamicState> dynamicStateEnables = {
			VK_DYNAMIC_STATE_VIEWPORT,
			VK_DYNAMIC_STATE_SCISSOR
		};
		VkPipelineDynamicStateCreateInfo dynamicState =
			vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);

		std::array<VkPipelineShaderStageCreateInfo, 2> shaderStages;

		VkGraphicsPipelineCreateInfo pipelineCreateInfo =
			vks::initializers::pipelineCreateInfo(pipelineLayouts.shared, renderPass, 0);

		pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
		pipelineCreateInfo.pRasterizationState = &rasterizationState;
		pipelineCreateInfo.pColorBlendState = &colorBlendState;
		pipelineCreateInfo.pMultisampleState = &multisampleState;
		pipelineCreateInfo.pViewportState = &viewportState;
		pipelineCreateInfo.pDepthStencilState = &depthStencilState;
		pipelineCreateInfo.pDynamicState = &dynamicState;
		pipelineCreateInfo.stageCount = shaderStages.size();
		pipelineCreateInfo.pStages = shaderStages.data();

		// Shadow map cascade debug quad display
		rasterizationState.cullMode = VK_CULL_MODE_NONE;
		shaderStages[0] = loadShader(getAssetPath() + "shaders/shadowmappingcascade/debugshadowmap.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/shadowmappingcascade/debugshadowmap.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		// Empty vertex input state
		VkPipelineVertexInputStateCreateInfo emptyInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		pipelineCreateInfo.pVertexInputState = &emptyInputState;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.debugShadowMap));

		// Vertex bindings and attributes
		std::vector<VkVertexInputBindingDescription> vertexInputBindings = {
			vks::initializers::vertexInputBindingDescription(0, vertexLayout.stride(), VK_VERTEX_INPUT_RATE_VERTEX)
		};

		std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
			vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0),					// Location 0: Position
			vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32_SFLOAT, sizeof(float) * 3),		// Location 1: UV
			vks::initializers::vertexInputAttributeDescription(0, 2, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 5),	// Location 2: Color
			vks::initializers::vertexInputAttributeDescription(0, 3, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 8)		// Location 3: Normal
		};

		VkPipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
		vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size());
		vertexInputState.pVertexBindingDescriptions = vertexInputBindings.data();
		vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
		vertexInputState.pVertexAttributeDescriptions = vertexInputAttributes.data();

		pipelineCreateInfo.pVertexInputState = &vertexInputState;

		// Scene rendering with shadows applied
		rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
		shaderStages[0] = loadShader(getAssetPath() + "shaders/shadowmappingcascade/scene.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/shadowmappingcascade/scene.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		// Use specialization constants to select between horizontal and vertical blur
		uint32_t enablePCF = 0;
		VkSpecializationMapEntry specializationMapEntry = vks::initializers::specializationMapEntry(0, 0, sizeof(uint32_t));
		VkSpecializationInfo specializationInfo = vks::initializers::specializationInfo(1, &specializationMapEntry, sizeof(uint32_t), &enablePCF);
		shaderStages[1].pSpecializationInfo = &specializationInfo;
		// No filtering
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.sceneShadow));
		// PCF filtering
		enablePCF = 1;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.sceneShadowPCF));

		// Shadow cascades colored debug visualzation
		shaderStages[1].pSpecializationInfo = nullptr;
		shaderStages[1] = loadShader(getAssetPath() + "shaders/shadowmappingcascade/debugcascades.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.debugCascades));

		// Shadow map generation pipeline
		shaderStages[0] = loadShader(getAssetPath() + "shaders/shadowmappingcascade/shadowmap.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
		shaderStages[1] = loadShader(getAssetPath() + "shaders/shadowmappingcascade/shadowmap.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
		// No blend attachment states (no color attachments used)
		colorBlendState.attachmentCount = 0;
		// Cull front faces
		depthStencilState.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		// Set depth bias (aka "Polygon offset"), required to avoid shadow mapping artefacts
		rasterizationState.depthBiasEnable = VK_TRUE;
		rasterizationState.depthBiasConstantFactor = depthBiasConstant;
		rasterizationState.depthBiasSlopeFactor = depthBiasSlope;
		rasterizationState.depthBiasClamp = 0.0f;
		pipelineCreateInfo.layout = pipelineLayouts.offscreen;
		pipelineCreateInfo.renderPass = shadowPass.renderPass;
		VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipelines.shadowMap));
	}

	// Prepare and initialize uniform buffer containing shader uniforms
	void prepareUniformBuffers()
	{
		// Offscreen vertex shader uniform buffer block
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&uniformBuffers.offscreen,
			sizeof(uboOffscreenVS)));

		// Scene uniform buffer blocks
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&uniformBuffers.scene,
			sizeof(uboVSscene)));
		VK_CHECK_RESULT(vulkanDevice->createBuffer(
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
			&uniformBuffers.sceneFS,
			sizeof(uboFSscene)));

		// Map persistent
		VK_CHECK_RESULT(uniformBuffers.offscreen.map());
		VK_CHECK_RESULT(uniformBuffers.scene.map());
		VK_CHECK_RESULT(uniformBuffers.sceneFS.map());

		updateLight();
		updateUniformBufferOffscreen();
		updateUniformBuffers();
	}

	// Calculate frustum split depths for the shadow cascades
	void updateSplitDepths()
	{
		float znear = camera.getNearClip();
		float zfar = camera.getFarClip();

		float i_f = 1.0f;
		float numCascades = (float)SHADOW_MAP_CASCADE_COUNT;
		for (uint32_t i = 0; i < SHADOW_MAP_CASCADE_COUNT - 1; i++, i_f += 1.0f) {			
			splitDepths[i] = glm::mix(
				znear + (i_f / numCascades)*(zfar - znear),
				znear * powf(zfar / znear, i_f / numCascades),
				SPLIT_LOG_FACTOR);
		}
		splitDepths[SHADOW_MAP_CASCADE_COUNT - 1] = zfar;
	}

	void updateLight()
	{
		lightPos = glm::vec3(-50.0f, -30.0f, -50.0f);
		// Animate the light source
		//lightPos.x = cos(glm::radians(timer * 360.0f)) * 50.0f;
		//lightPos.y = -50.0f + sin(glm::radians(timer * 360.0f)) * 20.0f;
		//lightPos.z = sin(glm::radians(timer * 360.0f)) * 50.0f;
	}

	void updateUniformBuffers()
	{
		// 3D scene
		uboVSscene.projection = glm::perspective(glm::radians(45.0f), (float)width / (float)height, zNear, zFar);

		uboVSscene.view = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, zoom));
		uboVSscene.view = glm::rotate(uboVSscene.view, glm::radians(rotation.x), glm::vec3(1.0f, 0.0f, 0.0f));
		uboVSscene.view = glm::rotate(uboVSscene.view, glm::radians(rotation.y), glm::vec3(0.0f, 1.0f, 0.0f));
		uboVSscene.view = glm::rotate(uboVSscene.view, glm::radians(rotation.z), glm::vec3(0.0f, 0.0f, 1.0f));

		uboVSscene.model = glm::mat4(1.0f);

		uboVSscene.projection = camera.matrices.perspective;
		uboVSscene.view = camera.matrices.view;
		uboVSscene.model = glm::mat4(1.0f);

		uboVSscene.lightPos = lightPos;

		uboVSscene.depthBiasMVP = uboOffscreenVS.depthMVP[0];
		memcpy(uniformBuffers.scene.mapped, &uboVSscene, sizeof(uboVSscene));

		for (uint32_t i = 0; i < SHADOW_MAP_CASCADE_COUNT; i++) {
			uboFSscene.cascadeSplits[i] = cascades[i].splitDepth;
		}
		memcpy(uniformBuffers.sceneFS.mapped, &uboFSscene, sizeof(uboFSscene));
	}

	void updateUniformBufferOffscreen()
	{
		updateSplitDepths();

		// Matrix from light's point of view
		glm::mat4 depthProjectionMatrix = glm::perspective(glm::radians(lightFOV), 1.0f, zNear, zFar);
		glm::mat4 depthViewMatrix = glm::lookAt(lightPos, glm::vec3(0.0f), glm::vec3(0, 1, 0));
		glm::mat4 depthModelMatrix = glm::mat4(1.0f);

		uboOffscreenVS.depthMVP[0] = depthProjectionMatrix * depthViewMatrix * depthModelMatrix;
		for (uint32_t i = 0; i < SHADOW_MAP_CASCADE_COUNT; i++) {
			uboOffscreenVS.depthMVP[i] = cascades[i].viewProjMatrix;
		}

		memcpy(uniformBuffers.offscreen.mapped, &uboOffscreenVS, sizeof(uboOffscreenVS));
	}

	void draw()
	{
		VulkanExampleBase::prepareFrame();

		// Shadow map generation
		submitInfo.pWaitSemaphores = &semaphores.presentComplete;
		submitInfo.pSignalSemaphores = &shadowPass.semaphore;

		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &shadowPass.commandBuffer;
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

		// Scene rendering
		submitInfo.pWaitSemaphores = &shadowPass.semaphore;;
		submitInfo.pSignalSemaphores = &semaphores.renderComplete;
		submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));

		VulkanExampleBase::submitFrame();
	}

	void prepare()
	{
		VulkanExampleBase::prepare();
		loadAssets();
		prepareOffscreenFramebuffer();
		prepareUniformBuffers();
		setupDescriptorSetLayout();
		preparePipelines();
		setupDescriptorPool();
		setupDescriptorSets();
		buildCommandBuffers();
		buildOffscreenCommandBuffer();
		prepared = true;
	}

	virtual void render()
	{
		if (!prepared)
			return;
		draw();
		if (!paused)
		{
			updateLight();
			updateUniformBufferOffscreen();
			updateUniformBuffers();
		}
	}

	virtual void viewChanged()
	{
		updateUniformBufferOffscreen();
		updateUniformBuffers();
	}

	virtual void OnUpdateUIOverlay(vks::UIOverlay *overlay)
	{
		if (overlay->header("Settings")) {
			if (overlay->comboBox("Scenes", &sceneIndex, sceneNames)) {
				buildCommandBuffers();
				buildOffscreenCommandBuffer();
			}
			if (overlay->checkBox("Display cascades", &displayCascades)) {
				buildCommandBuffers();
			}
			if (overlay->checkBox("Display shadow map", &displayShadowMap)) {
				buildCommandBuffers();
			}
			if (displayShadowMap) {
				if (overlay->sliderInt("Cascade", &displayShadowMapCascadeIndex, 0, SHADOW_MAP_CASCADE_COUNT - 1)) {
					buildCommandBuffers();
				}
			}
			if (overlay->checkBox("PCF filtering", &filterPCF)) {
				buildCommandBuffers();
			}
		}
	}
};

VULKAN_EXAMPLE_MAIN()
