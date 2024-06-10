import streamlit as st
from chatbot import predict_class, get_response, intents

# Título de la página
st.title("Asistente virtual")

# Inicialización de 'messages' si aún no existe
if "messages" not in st.session_state:
    st.session_state.messages = []

# Inicialización de 'first_message' si aún no existe
if "first_message" not in st.session_state:
    st.session_state.first_message = True

# Mostrar mensajes previos
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Mensaje inicial del asistente
if st.session_state.first_message:
    with st.chat_message("assistant"):
        st.markdown("Hola, ¿cómo puedo ayudarte?")
    st.session_state.messages.append({"role": "assistant", "content": "Hola, ¿Cómo puedo ayudarte?"})
    st.session_state.first_message = False

# Entrada del usuario
if prompt := st.chat_input("¿Cómo puedo ayudarte?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})


    
    #Implementacion del algoritmo de IA
    insts = predict_class(prompt)
    res = get_response(insts, intents)
    
    with st.chat_message("assistant"):
        st.markdown(res)
    st.session_state.messages.append({"role": "assistant", "content": res})
    
    
