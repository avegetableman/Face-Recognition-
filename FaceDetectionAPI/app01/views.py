from django.http import HttpResponse
import sqlite3
import sqlite_vec #更具需求选择向量数据库数据多选milvus等大型向量数据库
import numpy as np
import cv2
from sqlite_vec import serialize_float32
from AI_model.utils_for_onnx import NMS,Anchors,transform_box,normalize_image
import onnxruntime as ort
from AI_model.ort_inference import crop,resize
from PIL import Image
from io import BytesIO
#按cpu服务器处理
path = r'AI_model/detect.onnx'
session = ort.InferenceSession(path)
face_path=r'AI_model/face.onnx'
a= ort.InferenceSession('face.onnx')

filter = NMS()
db = sqlite3.connect('vec_database.db')
db.enable_load_extension(1)
sqlite_vec.load(db)
db.enable_load_extension(0)
def find_vectors_within_distance(db_conn, query_vector, max_distance=1):
    query_vector_blob = serialize_float32(query_vector)
    
    query = '''
    SELECT id, vec_distance_L2(vector, ?) AS distance
    FROM vector
    WHERE vec_distance_L2(vector, ?) < ?
    '''
    
    cursor = db_conn.execute(query, (query_vector_blob, query_vector_blob, max_distance))
    results = cursor.fetchall()
    return results

# Create your views here.
def regest(request):
    if request.method == 'POST':
        input_name = session.get_inputs()[0].name
        output_names = [session.get_outputs()[0].name, session.get_outputs()[1].name]
        data = request.FILES['image']
        pil_image = Image.open(BytesIO(data.read()))
        image = np.array(pil_image).astype(np.float32)/255.0
        data0 = crop(image)
        data0 = resize(data0)
        e = np.expand_dims(data0, axis=0)
        image_bchw = np.transpose(e, (0, 3, 1, 2))
        answer = session.run(output_names, {input_name: image_bchw})
        t = transform_box(Anchors)
        a = t.forward(answer,th=0.5)
        annn = filter.forward(a)
        if len(annn) > 1:
            return HttpResponse('face count > 1!')
        else:
            x,y,w,h = annn[0]['box']
            new_data0 = data0[int(y):int(y+h),int(x):int(x+w)]
            convert=cv2.cvtColor(new_data0, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(convert)
            squre = max(new_data0.shape[0],new_data0.shape[1])
            background=Image.new(mode='RGB',size=(squre,squre),color=(0,0,0))
            if new_data0.shape[0] > new_data0.shape[1]:
                position = (squre//2-new_data0.shape[1]//2,0)
            else:
                position = (0,squre//2-new_data0.shape[0]//2)
            background.paste(pil,position)
            buffer = BytesIO()
            background.save(buffer, format='JPEG')  # 可以使用 'PNG' 或其他格式
            buffer.seek(0)

            # 从字节流读取图像数据
            buffer_bytes = buffer.getvalue()
            nparr = np.frombuffer(buffer_bytes, np.uint8)
            target_picture = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            face_input_name = a.get_inputs()[0].name
            face_output_names = [a.get_outputs()[0].name]
            face_data = cv2.resize(target_picture, (130,130))
            face_data = normalize_image(face_data)
            face_data = np.expand_dims(face_data, axis=0)
            face_data = np.transpose(face_data,(0, 3, 1, 2))
            face_answer = a.run(face_output_names, {face_input_name: face_data})
            db.execute('''
            insert into vector (vector) values (?)
            ''',[serialize_float32(face_answer)])           
            return HttpResponse('ok')

def login(request):
    if request.method == 'POST':
        input_name = session.get_inputs()[0].name
        output_names = [session.get_outputs()[0].name, session.get_outputs()[1].name]
        data = request.FILES['image']
        pil_image = Image.open(BytesIO(data.read()))
        image = np.array(pil_image).astype(np.float32)/255.0
        data0 = crop(image)
        data0 = resize(data0)
        e = np.expand_dims(data0, axis=0)
        image_bchw = np.transpose(e, (0, 3, 1, 2))
        answer = session.run(output_names, {input_name: image_bchw})
        t = transform_box(Anchors)
        a = t.forward(answer,th=0.5)
        annn = filter.forward(a)
        faces = []
        for a in annn:
            x,y,w,h = a['box']
            new_data0 = data0[int(y):int(y+h),int(x):int(x+w)]
            faces.append(new_data0)
        for face in faces:
            convert=cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(convert)
            squre = max(face.shape[0],face.shape[1])
            background=Image.new(mode='RGB',size=(squre,squre),color=(0,0,0))
            if face.shape[0] > face.shape[1]:
                position = (squre//2-face.shape[1]//2,0)
            else:
                position = (0,squre//2-face.shape[0]//2)
            background.paste(pil,position)
            buffer = BytesIO()
            background.save(buffer, format='JPEG') 
            buffer.seek(0)

            # 从字节流读取图像数据
            buffer_bytes = buffer.getvalue()
            nparr = np.frombuffer(buffer_bytes, np.uint8)
            target_picture = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            face_input_name = a.get_inputs()[0].name
            face_output_names = [a.get_outputs()[0].name]
            face_data = cv2.resize(target_picture, (130,130))
            face_data = normalize_image(face_data)
            face_data = np.expand_dims(face_data, axis=0)
            face_data = np.transpose(face_data,(0, 3, 1, 2))
            face_answer = a.run(face_output_names, {face_input_name: face_data})
            similar_vectors = find_vectors_within_distance(db, face_answer)
            vector_id, _ = similar_vectors
            answ = {}
            answ['id'] = vector_id        
            return HttpResponse(answ)
    return