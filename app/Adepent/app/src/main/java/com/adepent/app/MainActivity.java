package com.adepent.app;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Base64;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.github.dhaval2404.imagepicker.ImagePicker;

import org.jetbrains.annotations.Nullable;
import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.FormBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class MainActivity extends AppCompatActivity {

    private Uri imgUri;
    private File imgFile;
    private String imgBase64;

    private Button btn_choose;
    private Button btn_upload;
    private TextView tv_result;
    private ImageView iv_show;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        getSupportActionBar().setSubtitle("Aplikasi Deteksi Penyakit Daun Teh");

        btn_choose = findViewById(R.id.btn_choose);
        btn_upload = findViewById(R.id.btn_upload);

        tv_result = findViewById(R.id.tv_result);
        iv_show = findViewById(R.id.iv_show);

        LoadingDialog loadingDialog = new LoadingDialog(MainActivity.this);

        btn_choose.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View view) {

                ImagePicker.with(MainActivity.this)
                        .cropSquare()	    			//Crop image(Optional), Check Customization for more option
                        .compress(1024)			//Final image size will be less than 1 MB(Optional)
                        .maxResultSize(1024, 1024)	//Final image resolution will be less than 1080 x 1080(Optional)
                        .start(100);

            }
        });

        btn_upload.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View view) {

                if(imgUri != null && imgBase64 != null) {

                    loadingDialog.startAlertLoadingDialog();

                    OkHttpClient okHttpClient = new OkHttpClient();

                    RequestBody requestBody = new FormBody.Builder().add("image", imgBase64).build();
//                    Request request = new Request.Builder().url("http://192.168.0.111:5000/predict")
//                            .post(requestBody).build();
                    Request request = new Request.Builder().url("http://192.168.244.1:5000/predict")
                            .post(requestBody).build();

                    okHttpClient.newCall(request).enqueue(new Callback() {
                        @Override
                        public void onFailure(@NonNull Call call, @NonNull IOException e) {

                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    loadingDialog.dismissAlertLoadingDialog();
                                    Toast.makeText(MainActivity.this, e.getMessage(), Toast.LENGTH_LONG).show();
                                }
                            });
                        }

                        @Override
                        public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                            runOnUiThread(new Runnable() {
                                @Override
                                public void run() {
                                    try {
                                        loadingDialog.dismissAlertLoadingDialog();

                                        JSONObject jsonObject = new JSONObject(response.body().string());

                                        String result = jsonObject.getString("result");
                                        String description = jsonObject.getString("description");

                                        Intent intent = new Intent(MainActivity.this, ResultActivity.class);
                                        intent.putExtra("result", result);
                                        intent.putExtra("description", description);
                                        intent.putExtra("imgUri", imgUri.toString());

                                        Toast.makeText(MainActivity.this, jsonObject.getString("message"), Toast.LENGTH_LONG).show();

                                        startActivity(intent);

                                    } catch (IOException e) {
                                        throw new RuntimeException(e);
                                    } catch (JSONException e) {
                                        throw new RuntimeException(e);
                                    }


                                }
                            });
                        }
                    });

                } else {

                    Toast.makeText(MainActivity.this, "Pilih gambar terlebih dahulu!", Toast.LENGTH_LONG).show();

                }

            }
        });

    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.menu, menu);

        return true;
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        switch (item.getItemId()) {
            case R.id.menu_about:
                Intent intent = new Intent(MainActivity.this, AboutActivity.class);
                startActivity(intent);
                return true;
        }
        return super.onOptionsItemSelected(item);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {

        super.onActivityResult(requestCode, resultCode, data);

        if(requestCode == 100 && resultCode == RESULT_OK && data != null) {

            imgUri = data.getData();
            imgFile = new File(imgUri.getPath());

            iv_show.setImageURI(imgUri);
            iv_show.getLayoutParams().height = 500;
            iv_show.getLayoutParams().width = 500;

            tv_result.setText(imgFile.getName());

            try {

                // Inisialisasi bitmap
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(getContentResolver()
                        , imgUri);

                // Inisialisasi bytestream
                ByteArrayOutputStream stream = new ByteArrayOutputStream();

                // Kompres bitmap
                bitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream);

                // Inisialisasi byte array
                byte[] bytes = stream.toByteArray();

                // Peroleh base64 encoded string
                imgBase64 = Base64.encodeToString(bytes, Base64.DEFAULT);

            } catch (IOException e) {

                throw new RuntimeException(e);
            }

        }


    }
}