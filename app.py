import pygame
import PIL as image
from cnn import CNN, predict_image
import torch
import torchvision.transforms as transforms
import torchvision.datasets
from PIL import Image

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN()
model.load_state_dict(torch.load('cnn.pth'))
model.eval()

def app():
    pygame.init()

    fps = 60
    timer = pygame.time.Clock()
    width = 800
    height = 600
    active_size = 0
    painting = []
    font = pygame.font.Font(None, 40)
    prediction = ""

    screen = pygame.display.set_mode([width, height])
    pygame.display.set_caption('Number Predictor-Paint App')

    def draw_menu():
        pygame.draw.rect(screen, 'gray',[width - 100, 0,100,height])
        pygame.draw.line(screen, 'black', (width - 100, 0), (width - 100, height), 3)

        l_brush = pygame.draw.rect(screen, 'black', [width - 72.5,  20, 50,50])
        pygame.draw.circle(screen, 'white', (width - 47.5, 45), 22)
        m_brush = pygame.draw.rect(screen, 'black', [width - 72.5, 100, 50,50])
        pygame.draw.circle(screen, 'white', (width - 47.5, 125), 15)
        s_brush = pygame.draw.rect(screen, 'black', [width - 72.5, 180, 50,50])
        pygame.draw.circle(screen, 'white', (width - 47.5, 205), 7)
        brush_list = [l_brush, m_brush, s_brush]

        # save button
        save_button = pygame.draw.rect(screen, 'black',[width - 72.5, 320, 50,30])
        return brush_list, save_button

    def draw_painting(paints):
        for i in range(len(paints)):
            pygame.draw.circle(screen, paints[i][0], paints[i][1], paints[i][2])

    run = True
    while run:
        timer.tick(fps)
        screen.fill('black')

        mouse = pygame.mouse.get_pos()
        brushes, save_butt = draw_menu()
        left_click = pygame.mouse.get_pressed()[0]
        prediction_text = font.render(
                                    f"Prediction: {prediction}", True, "white"
                                     )
        screen.blit(prediction_text, (20, 20))

    # ====================================================================

        if mouse[0] < (width - 100):
            pygame.draw.circle(screen, 'white', (mouse), active_size)

            if left_click:
                painting.append(('white', mouse, active_size))

        draw_painting(paints=painting)


    # ====================================================================

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                for i in range(len(brushes)):
                    if brushes[i].collidepoint(event.pos):
                        active_size = 20 - (i * 5)

            if event.type == pygame.MOUSEBUTTONDOWN:
                if save_butt.collidepoint(event.pos):
                    drawing_area = screen.subsurface((0,0,width - 100,height))
                    pygame.image.save(drawing_area, 'drawing.png')

                    image = Image.open("drawing.png").convert("L")
                    transform = transforms.Compose([
                                                    transforms.Resize((28, 28)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.1307,), (0.3081,))
                                                    ])
                    image = transform(image)
                    prediction = predict_image(model, image)
                    painting = []

        pygame.display.flip()
    pygame.quit()