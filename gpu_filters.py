import glfw
from OpenGL.GL import *
import numpy as np
import cv2
import ctypes

VERTEX = """
#version 330
layout(location=0) in vec2 pos;
layout(location=1) in vec2 tex;
out vec2 uv;
void main(){
    uv = tex;
    gl_Position = vec4(pos,0,1);
}
"""

FRAGMENT = """
#version 330
out vec4 FragColor;
in vec2 uv;

uniform sampler2D image;
uniform vec2 texelSize;
uniform float sharpStrength;
uniform float contrastLevel;

void main(){
    vec3 color = texture(image, uv).rgb;

    // CONTRAST
    color = (color - 0.5) * contrastLevel + 0.5;

    // GAUSSIAN BLUR
    vec3 blur = vec3(0.0);
    vec2 off[9] = vec2[](
        vec2(-1,-1), vec2(0,-1), vec2(1,-1),
        vec2(-1,0),  vec2(0,0),  vec2(1,0),
        vec2(-1,1),  vec2(0,1),  vec2(1,1)
    );
    float k[9] = float[](1,2,1, 2,4,2, 1,2,1);

    for(int i=0;i<9;i++)
        blur += texture(image, uv + off[i]*texelSize).rgb * k[i];

    blur /= 16.0;

    // UNSHARP MASK
    vec3 sharp = color + sharpStrength * (color - blur);
    FragColor = vec4(sharp,1);
}
"""

class GPUProcessor:
    def __init__(self):
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        self.window = glfw.create_window(64,64,"",None,None)
        glfw.make_context_current(self.window)

        self.program = self._create_program()
        self._create_quad()

    def _compile(self, src, t):
        s = glCreateShader(t)
        glShaderSource(s, src)
        glCompileShader(s)
        return s

    def _create_program(self):
        vs = self._compile(VERTEX, GL_VERTEX_SHADER)
        fs = self._compile(FRAGMENT, GL_FRAGMENT_SHADER)
        p = glCreateProgram()
        glAttachShader(p, vs)
        glAttachShader(p, fs)
        glLinkProgram(p)
        return p

    def _create_quad(self):
        quad = np.array([
            -1,-1, 0,0,  1,-1, 1,0,  1, 1, 1,1,
            -1,-1, 0,0,  1, 1, 1,1, -1, 1, 0,1
        ], np.float32)

        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER,self.vbo)
        glBufferData(GL_ARRAY_BUFFER,quad.nbytes,quad,GL_STATIC_DRAW)
        glVertexAttribPointer(0,2,GL_FLOAT,False,16,None)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1,2,GL_FLOAT,False,16,ctypes.c_void_p(8))
        glEnableVertexAttribArray(1)

    def process(self, img, sharp=1.5, contrast=1.01, use_contrast=True, use_unsharp=True):
        h, w, _ = img.shape

        # Textura de origem
        srcTex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, srcTex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, img)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        # Textura de saída
        outTex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, outTex)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        # FBO
        fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, outTex, 0)

        glViewport(0, 0, w, h)
        glUseProgram(self.program)
        glUniform1f(glGetUniformLocation(self.program, "sharpStrength"), sharp)
        glUniform1f(glGetUniformLocation(self.program, "contrastLevel"), contrast)
        glUniform2f(glGetUniformLocation(self.program, "texelSize"), 1.0 / w, 1.0 / h)
        glUniform1i(glGetUniformLocation(self.program, "use_contrast"), int(use_contrast))
        glUniform1i(glGetUniformLocation(self.program, "use_unsharp"), int(use_unsharp))

        glBindTexture(GL_TEXTURE_2D, srcTex)
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 6)

        # Ler pixels e inverter verticalmente
        data = glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)
        out = np.frombuffer(data, np.uint8).reshape(h, w, 3)

        # Limpeza
        glDeleteTextures([srcTex, outTex])
        glDeleteFramebuffers(1, [fbo])

        return out

