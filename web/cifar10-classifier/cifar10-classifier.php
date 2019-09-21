<?php
  $target_dir = "temp/";
  $target_file = $target_dir . basename($_FILES["userfile"]["name"]);
  $uploadOk = 1;
  $imageFileType = strtolower(pathinfo($target_file,PATHINFO_EXTENSION));
  // Check if image file is a actual image or fake image
  if(isset($_POST["submit"])) {
    $check = getimagesize($_FILES["userfile"]["tmp_name"]);
    if($check !== false) {
      //echo "File is an image - " . $check["mime"] . ".<br>";
      $uploadOk = 1;
    }
    else
    {
      echo "File is not an image.<br>";
      $uploadOk = 0;
    }
}

// Check if file already exists
if (file_exists($target_file)) {
    echo "Sorry, file already exists.<br>";
    $uploadOk = 0;
}
// Check file size
if ($_FILES["userfile"]["size"] > 2000000) {
    echo "Sorry, your file is too large.<br>";
    $uploadOk = 0;
}
// Allow certain file formats
if($imageFileType != "jpg" && $imageFileType != "png" && $imageFileType != "jpeg"
&& $imageFileType != "gif" ) {
    echo "Sorry, only JPG, JPEG, PNG & GIF files are allowed.<br>";
    $uploadOk = 0;
}
// Check if $uploadOk is set to 0 by an error
if ($uploadOk == 0) {
    echo "Sorry, your file was not uploaded.<br>";
// if everything is ok, try to upload file
} else {
    if (move_uploaded_file($_FILES["userfile"]["tmp_name"], $target_file)) {
        //echo "The file ". basename( $_FILES["userfile"]["name"]). " is being processed.<br>";
    } else {
        echo "Sorry, there was an error uploading your file.<br>";
    }
}

if ($uploadOk == 1){
    if (file_exists($target_file)) {
        $python = "/usr/bin/python3";
        $script = "cifar10-classifier.py";
        $command = $python . ' ' . $script . ' ' . $target_file;
        echo exec($command)

    }else {
        echo "Sorry, there was an error processing your file.<br>";
    }
}

?> 
 
