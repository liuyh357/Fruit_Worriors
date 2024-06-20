function uploadFile() {
  const fileInput = document.getElementById('upload');
  
  if (fileInput.files.length === 0) {
      alert("请选择一个文件!");
      return;
  }
  
  const file = fileInput.files[0];
  
  const formData = new FormData();
  
  formData.append('file', file);
  
  fetch('http://172.16.91.233:8000/upload', {
      method: 'POST',
      body: formData
  })
  .then(response => response.json())
  .then(data => {
      if (data.success) {
          const base64Image = 'data:image/jpg;base64,' + data.image_base64;
          document.getElementById('fruit-image').src = base64Image;
      } else {
          alert("告诉你个秘密，我能看见鬼.");
      }
  })
  .catch(error => {
      console.error('Error:', error);
  });
}
